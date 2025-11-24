# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .msdeform import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .adapter_modules_bitemporal import (InteractionBlock, SpatialPriorModule,
                              deform_inputs)
from .vit import TIMMVisionTransformer

_logger = logging.getLogger(__name__)


class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, freeze_vit=False, *args, **kwargs):

        super().__init__(num_heads=num_heads, pretrained=pretrained,
                         with_cp=with_cp, *args, **kwargs)
        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.diff_norm1 = nn.SyncBatchNorm(embed_dim)
        self.diff_norm2 = nn.SyncBatchNorm(embed_dim)
        self.diff_norm3 = nn.SyncBatchNorm(embed_dim)
        self.diff_norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x1, x2):
        # Get reference points for deformable attention
        deform_inputs1, deform_inputs2 = deform_inputs(x1)

        # SPM forward
        c1_1, c2_1, c3_1, c4_1 = self.spm(x1)
        c1_2, c2_2, c3_2, c4_2 = self.spm(x2)
        
        
        c1, c2, c3, c4 =c1_1 - c1_2, c2_1 - c2_2, c3_1 - c3_2, c4_1 - c4_2 # Difference-based Bitemporal Fusion
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x1, H, W = self.patch_embed(x1)
        x2, _, _ = self.patch_embed(x2)
        bs, n, dim = x1.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x1 = self.pos_drop(x1 + pos_embed)
        x2 = self.pos_drop(x2 + pos_embed)

        # Interaction
        outs_x1, outs_x2, outs_c = list(), list(), list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # c is the F_{sp}
            x1, x2 , c = layer(x1, x2, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs_x1.append(x1.transpose(1, 2).view(bs, dim, H, W).contiguous())
            outs_x2.append(x2.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape

        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1
        f1 = self.diff_norm1(c1)
        f2 = self.diff_norm2(c2)
        f3 = self.diff_norm3(c3)
        f4 = self.diff_norm4(c4)

        CD_Feature=[f1,f2,f3,f4]

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs_x1
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        x1_features_seg=[f1,f2,f3,f4]

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs_x1
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        x2_features_seg=[f1,f2,f3,f4]

        # Final Norm
        return CD_Feature, x1_features_seg, x2_features_seg

if __name__ == '__main__':
    model = ViTAdapter(
        pretrain_size=224,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        conv_inplane=64,
        deform_num_heads=6,
        n_points=4,
        init_values=0.,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        with_cp=False
    )
    model.cuda()
    _logger.info(model)

    input = torch.randn(8, 3, 224, 224).cuda()
    output = model(input,input)
    for o in output:
        for feature in o:
            print(feature.shape)

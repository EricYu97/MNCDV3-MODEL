from .pretrained_backbone.ViT_Adapter_Bitemporal import ViTAdapter
from .decoder.upernet import SemanticSegmentationHead, ChangeDetectionHead
import torch.nn as nn
from thop import profile
import torch 
from .pretrained_backbone.ViT_Adapter import ViTAdapter

class MNCDV3_Model(nn.Module):
    def __init__(self, **kwargs):
        super(MNCDV3_Model, self).__init__()

        self.backbone = ViTAdapter(
        pretrain_size=224,
        img_size=224,
        patch_size=16,
        in_chans=9,
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
        with_cp=False,
        **kwargs
        )
        self.CD_decoder=ChangeDetectionHead(num_labels=2, embed_dims=768, img_size=224, fusion_strategy='diff')
        self.seg_decoder=SemanticSegmentationHead(num_labels=6, embed_dims=768, img_size=224)


    def forward(self, x1, x2, x1_label=None, x2_label=None, change_label=None):

        x1_features_seg, x2_features_seg=self.backbone(x1), self.backbone(x2)
        CD_Feature=[torch.abs(f1 - f2) for f1, f2 in zip(x1_features_seg, x2_features_seg)]  # Difference-based CD Feature

        # CD_Feature, x1_features_seg, x2_features_seg = self.backbone(x1, x2)

        cd_outputs = self.CD_decoder(CD_Feature, labels=change_label)
        x1_seg_outputs = self.seg_decoder(x1_features_seg, labels=x1_label)
        x2_seg_outputs = self.seg_decoder(x2_features_seg, labels=x2_label)

        # return CD_Feature, x1_features_seg, x2_features_seg
        return cd_outputs, x1_seg_outputs, x2_seg_outputs
    
if __name__ == '__main__':
    def count_model_parameters(model):
        """
        Calculates the total, trainable, and frozen parameters of a PyTorch model.
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for name, param in model.named_parameters():
            num_param = param.nelement()
            total_params += num_param
            
            if param.requires_grad:
                trainable_params += num_param
            else:
                frozen_params += num_param
        
        return {
            "Total Params (Inference)": total_params,
            "Trainable Params (Training)": trainable_params,
            "Frozen Params (Training)": frozen_params
        }

    model = MNCDV3_Model().cuda()
    input1 = torch.randn(2, 3, 224, 224).cuda()
    input2 = torch.randn(2, 3, 224, 224).cuda()
    cd_outputs, x1_seg_outputs, x2_seg_outputs = model(input1, input2)
    print(cd_outputs.logits.shape, x1_seg_outputs.logits.shape, x2_seg_outputs.logits.shape)

    macs, total_params = profile(model, inputs=(input1, input2, ), verbose=False)

    # Convert MACs (Multiply-Accumulate Operations) to FLOPs (typically FLOPs ≈ 2 * MACs)
    # Note: thop typically reports MACs, often used interchangeably with FLOPs in some literature.
    # For a more precise FLOPs count, you often multiply MACs by 2.
    GFLOPs = macs / 1e9 
    MParams = total_params / 1e6

    print(f"--- Inference Cost (FLOPs & Total Parameters) ---")
    print(f"Total Parameters: {MParams:.2f} M")
    print(f"MACs (Multiply-Accumulate Operations): {macs / 1e6:.2f} M")
    print(f"Estimated FLOPs (approx. 2x MACs): {GFLOPs * 2:.2f} G")

    param_counts = count_model_parameters(model)
    print("\n--- Parameter Analysis (Training vs. Inference) ---")
    print(f"Total Parameters (Inference Cost): {param_counts['Total Params (Inference)']:_} (used in forward pass)")
    print(f"Trainable Parameters (Training Cost): {param_counts['Trainable Params (Training)']:_} (updated by optimizer)")
    print(f"Frozen Parameters: {param_counts['Frozen Params (Training)']:_} (not updated)")
    # print(CD_Feature[0].shape, CD_Feature[1].shape, CD_Feature[2].shape, CD_Feature[3].shape)
import torch
from tqdm import tqdm
from model.model import MNCDV3_Model
from dataloader import MNCDV3_Dataset
import os
from torchmetrics import F1Score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

def train(args):
    # Initialize distributed training
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    epoch = 20
    is_main_process = rank == 0

    train_dataset = MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='train')
    val_dataset = MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='val')
    test_dataset = MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='test')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=16)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, sampler=val_sampler, num_workers=16)

    if is_main_process:
        print(f"Dataset and Dataloader prepared for training {len(train_dataset)} samples and validating {len(val_dataset)} samples and testing {len(test_dataset)} samples.")

    model = MNCDV3_Model().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    for ep in range(epoch):
        train_sampler.set_epoch(ep)
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device, is_main_process)
        if is_main_process:
            print(f"Epoch {ep+1}/{epoch}, Loss: {epoch_loss:.4f}")
        if ep % 2 == 1:
            evaluate_model(model, val_dataloader, ep+1, device, is_main_process, save_suffix='MNCDV3_Model')
    
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, scheduler, device, is_main_process):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(dataloader, disable=not is_main_process)):
        pre_image, post_image, pre_label, post_label, change_label = batch
        
        pre_image = pre_image.to(device)
        post_image = post_image.to(device)
        pre_label = pre_label.to(device)
        post_label = post_label.to(device)
        change_label = change_label.to(device)

        optimizer.zero_grad()

        cd_outputs, x1_seg_outputs, x2_seg_outputs = model(x1=pre_image, x2=post_image, change_label=change_label, x1_label=pre_label, x2_label=post_label)

        cd_loss = cd_outputs.loss
        x1_seg_loss = x1_seg_outputs.loss
        x2_seg_loss = x2_seg_outputs.loss

        loss = cd_loss + x1_seg_loss + x2_seg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if i % 50 == 49 and is_main_process:
            print(f"  Batch {i+1}, Total Loss: {loss.item():.4f}, Seg_Loss: {x1_seg_loss.item()+x2_seg_loss.item():.4f}, CD_Loss: {cd_loss.item():.4f}")
    
    epoch_loss = running_loss / (i + 1)
    return epoch_loss

def evaluate_model(model, val_dataloader, epoch, device, is_main_process, save_suffix):
    model.eval()
    
    b_f1 = F1Score(task='multiclass', num_classes=6).to(device)
    m_f1 = F1Score(task='multiclass', num_classes=2).to(device)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader, disable=not is_main_process)):
            pre_image, post_image, pre_label, post_label, change_label = batch
            
            pre_image = pre_image.to(device)
            post_image = post_image.to(device)
            pre_label = pre_label.to(device)
            post_label = post_label.to(device)

            outputs = model(x1=pre_image, x2=post_image)

            cd_logits = None
            x1_logits = None
            x2_logits = None
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                cd_out, x1_out, x2_out = outputs[:3]
                cd_logits = getattr(cd_out, "logits", cd_out)
                x1_logits = getattr(x1_out, "logits", x1_out)
                x2_logits = getattr(x2_out, "logits", x2_out)
            elif isinstance(outputs, dict):
                cd_logits = outputs.get("cd_logits") or outputs.get("cd_preds") or outputs.get("cd")
                seg_preds = outputs.get("seg_preds") or outputs.get("seg_logits") or outputs.get("seg")
                if isinstance(seg_preds, (list, tuple)) and len(seg_preds) >= 2:
                    x1_logits, x2_logits = seg_preds[0], seg_preds[1]
                elif isinstance(seg_preds, torch.Tensor):
                    b = seg_preds.shape[0]
                    half = b // 2
                    x1_logits, x2_logits = seg_preds[:half], seg_preds[half:half*2]

            if x1_logits is None or x2_logits is None or cd_logits is None:
                continue

            x1_pred = torch.argmax(x1_logits, dim=1)
            x2_pred = torch.argmax(x2_logits, dim=1)

            b_f1.update(x1_pred.flatten(), pre_label.flatten())
            b_f1.update(x2_pred.flatten(), post_label.flatten())

            cd_true = (post_label != pre_label).long()
            cd_pred = torch.argmax(cd_logits, dim=1)
            m_f1.update(cd_pred.flatten(), cd_true.flatten())

    cd_f1 = m_f1.compute()
    seg_f1 = b_f1.compute()

    if is_main_process:
        print(f"Evaluation for Epoch {epoch} Completed, Seg_F1: {seg_f1}, CD_F1: {cd_f1}")

        save_pretrained_path = f"./exp/{save_suffix}/"
        os.makedirs(os.path.dirname(save_pretrained_path), exist_ok=True)
        torch.save(model.module.state_dict(), f'{save_pretrained_path}/{epoch}.pth')
        print(f"Saved model to {save_pretrained_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        default="/bigdata/3dabc/MNCD/MNCDV3_Bitemporal_Cropped_Size224_Step112",
        help="dataset root path"
    )
    args = parser.parse_args()
    train(args)
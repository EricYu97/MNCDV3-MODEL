import torch
from tqdm import tqdm
from model.model import MNCDV3_Model
from dataloader import MNCDV3_Dataset
import os
from torchmetrics import F1Score

from accelerate import DistributedDataParallelKwargs, Accelerator
import argparse
import torch.nn.functional as F

def train(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    epoch=100

    train_dataset=MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='train')
    val_dataset=MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='val')
    test_dataset=MNCDV3_Dataset(root_path=args.root_path, normalization=True, mode='test')

    train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
    val_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16)

    print(f"Dataset and Dataloader prepared for training {len(train_dataset)} samples and validating {len(val_dataset)} samples and testing {len(test_dataset)} samples.")

    model=MNCDV3_Model()
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    train_dataloader, val_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, val_dataloader, model, optimizer, scheduler)

    for ep in range(epoch):
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, accelerator)
        print(f"Epoch {ep+1}/{epoch}, Loss: {epoch_loss:.4f}")
        # if ep % 2 == 1:
        if True:
            evaluate_model(model, val_dataloader, ep+1, accelerator, save_suffix='MNCDV3_Model')

def train_one_epoch(model, dataloader, optimizer, scheduler, accelerator):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(dataloader)):
        pre_image, post_image, pre_label, post_label, change_label = batch

        optimizer.zero_grad()

        # outputs = model(pre_image, post_image)

        cd_outputs, x1_seg_outputs, x2_seg_outputs = model(x1=pre_image, x2=post_image, change_label=change_label, x1_label=pre_label, x2_label=post_label)

        cd_loss = cd_outputs.loss
        x1_seg_loss = x1_seg_outputs.loss
        x2_seg_loss = x2_seg_outputs.loss

        # Baseline: sum all losses
        # loss = cd_loss + x1_seg_loss + x2_seg_loss

        loss=x1_seg_loss+x2_seg_loss

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if i % 20 == 19:
            print(f"  Batch {i+1}, Total Loss: {loss.item():.4f}, Seg_Loss: {x1_seg_loss.item()+x2_seg_loss.item():.4f}, CD_Loss: {cd_loss.item():.4f}")
    epoch_loss = running_loss/(i+1)
    return epoch_loss

def evaluate_model(model, val_dataloader, epoch, accelerator, save_suffix):
    """
    Updated evaluate_model to match the training flow:
    - Expects model and val_dataloader to have been prepared by accelerator.prepare(...) if needed.
    - Uses the same batch structure as train(): (pre_image, post_image, pre_label, post_label, change_label)
    - Computes per-class F1 for segmentation (both frames) and change detection.
    """
    model.eval()
    device = accelerator.device
    num_classes_cd = 2  # Assuming binary change detection; modify if needed.
    # Metrics on device
    b_f1 = F1Score(task='multiclass', num_classes=6, average=None).to(device)
    m_f1 = F1Score(task='multiclass', num_classes=2, average=None).to(device)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader, disable=not accelerator.is_local_main_process)):
            # Expect same batch structure as training
            pre_image, post_image, pre_label, post_label, change_label = batch

            # If dataloader/model were prepared by accelerator, tensors may already be on the correct device.
            pre_image = pre_image.to(device)
            post_image = post_image.to(device)
            pre_label = pre_label.to(device)
            post_label = post_label.to(device)

            # Forward pass without labels -> get logits/predictions
            outputs = model(x1=pre_image, x2=post_image)

            # Support models that return (cd_out, x1_out, x2_out) where each may be an object with .logits
            cd_logits = None
            x1_logits = None
            x2_logits = None
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                cd_out, x1_out, x2_out = outputs[:3]
                cd_logits = getattr(cd_out, "logits", cd_out)
                x1_logits = getattr(x1_out, "logits", x1_out)
                x2_logits = getattr(x2_out, "logits", x2_out)

            cd_logits=F.softmax(cd_logits, dim=1) if cd_logits is not None else None
            x1_logits=F.softmax(x1_logits, dim=1) if x1_logits is not None else None  
            x2_logits=F.softmax(x2_logits, dim=1) if x2_logits is not None else None

            # If any logits still None, skip this batch (safer than crashing)
            if x1_logits is None or x2_logits is None or cd_logits is None:
                continue

            # Predictions
            x1_pred = torch.argmax(x1_logits, dim=1)
            x2_pred = torch.argmax(x2_logits, dim=1)

            # Update segmentation F1 with both frames
            b_f1.update(x1_pred.flatten(), pre_label.flatten())
            b_f1.update(x2_pred.flatten(), post_label.flatten())

            cd_pred = torch.argmax(cd_logits, dim=1)
            m_f1.update(cd_pred.flatten(), change_label.flatten())
            # print("Uniques", torch.unique(cd_pred), torch.unique(x1_pred), torch.unique(x2_pred), torch.unique(change_label), torch.unique(pre_label), torch.unique(post_label))
            # print("Counts", torch.count_nonzero(cd_pred), torch.count_nonzero(x1_pred), torch.count_nonzero(x2_pred), torch.count_nonzero(change_label), torch.count_nonzero(pre_label), torch.count_nonzero(post_label))
            # print("Shape", x1_pred.shape, x2_pred.shape, cd_pred.shape, pre_label.shape, post_label.shape, change_label.shape)

    # Compute metrics (may return tensors moved to device)
    cd_f1 = m_f1.compute()
    seg_f1 = b_f1.compute()

    # Print results (only on main process)


    if accelerator.is_local_main_process:
        print(f"Evaluation for Epoch {epoch} Completed, Seg_F1: {seg_f1}, Averaged_Seg_F1: {sum(seg_f1)/len(seg_f1)},CD_F1: {cd_f1}")

        save_pretrained_path = f"./exp/{save_suffix}/"
        os.makedirs(os.path.dirname(save_pretrained_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), f'{save_pretrained_path}/{epoch}.pth')
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
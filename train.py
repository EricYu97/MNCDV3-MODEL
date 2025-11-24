import torch
from tqdm import tqdm
from model.model import MNCDV3_Model
import accelerate
from

def train():
    dataset=
    model=MNCDV3_Model()

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
"""
Author: Amit Pratap

Script to train a Pytorch model
Use use a simple CNN model to train on the MNIST dataset.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from utils import MODEL_DIR, train_loader
from tqdm import tqdm

device = "cpu"

epochs = 1

model = Net().to(device)
optimizer = optim.Adam(model.parameters())

model.train()

progress_bar = tqdm(total=len(train_loader), desc="Training", leave=True)

for epoch in range(1, epochs+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # progress_bar.set_postfix(loss=loss)
        progress_bar.update(1)
        # tqdm(batch_idx * len(data), len(train_loader.dataset))
        # print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), loss.item()), end="")

progress_bar.close()

torch.save(model.state_dict(), MODEL_DIR / "original_model.pt")


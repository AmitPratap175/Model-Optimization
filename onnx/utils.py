

import torch
import pathlib
from torchvision import datasets, transforms

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

MODEL_DIR = pathlib.Path("./onnx_models")
MODEL_DIR.mkdir(exist_ok=True)

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

train_dataset = datasets.MNIST('./data', train=True, download=True,transform=transform)
test_dataset = datasets.MNIST('./data', train=False,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, 32)
test_loader = torch.utils.data.DataLoader(test_dataset, 32)

import torch
import torch.nn.functional as F
import openvino as ov
from utils import test_loader, MODEL_DIR

def test_ov(compiled_model, data_loader):
    # compiled_model = ov.compile_model(model)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        output = torch.tensor(compiled_model(data)[0])
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    return 100. * correct / len(data_loader.dataset)

core = ov.Core()

ov_model_path = MODEL_DIR / f"openvino_ir.xml"
ov_model = core.compile_model(ov_model_path, "CPU") 

quantized_model_path = MODEL_DIR / f"quant_openvino_ir.xml"
quantized_model = core.compile_model(quantized_model_path, "CPU")

acc = test_ov(ov_model, test_loader)
print(f"Accuracy of original model: {acc}")

qacc = test_ov(quantized_model, test_loader)
print(f"Accuracy of quantized model: {qacc}")
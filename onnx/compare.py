
import onnx, onnxruntime
import torch
import torch.nn.functional as F
from utils import test_loader, MODEL_DIR, to_numpy

def test_onnx(model_name, data_loader):
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_name)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        output = ort_session.run(None, ort_inputs)[0]
        output = torch.from_numpy(output)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    return 100. * correct / len(data_loader.dataset)

acc = test_onnx(MODEL_DIR / "mnist_model.onnx", test_loader)
print(f"Accuracy of the original model is {acc}%")

qacc = test_onnx(MODEL_DIR / "mnist_model_quant.onnx", test_loader)
print(f"Accuracy of the quantized model is {qacc}%")

static_qacc = test_onnx(MODEL_DIR / "mnist_model_static_quant.onnx", test_loader)
print(f"Accuracy of the static quantized model is {static_qacc}%")

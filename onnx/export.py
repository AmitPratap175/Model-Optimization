"""
Author: Amit Pratap

Script to export the trained PyTorch model to onnx format
Use use a simple CNN model to train on the MNIST dataset.
"""

import torch
import torch.quantization
import torch.onnx
from model import Net
import onnx, onnxruntime
import numpy as np
from utils import MODEL_DIR, train_loader

device = "cpu"

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_DIR / "original_model.pt"))

x, _ = next(iter(train_loader))
torch.onnx.export(model,              
                  x,                         
                  MODEL_DIR / "mnist_model.onnx",  
                  export_params=True,        
                  opset_version=11,          
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})


torch_out = model(x)

onnx_model = onnx.load(MODEL_DIR / "mnist_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(MODEL_DIR / "mnist_model.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
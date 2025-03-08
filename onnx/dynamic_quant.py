"""
Author: Amit Pratap

Script to export the trained PyTorch model to onnx format
Use use a simple CNN model to train on the MNIST dataset.
"""

from onnxruntime.quantization import quantize_dynamic, QuantType
from utils import MODEL_DIR

model_fp32 = MODEL_DIR / "mnist_model_processed.onnx"
model_quant = MODEL_DIR / "mnist_model_quant.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
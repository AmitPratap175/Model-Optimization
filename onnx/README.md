To quantize the model:
```bash
python -m onnxruntime.quantization.preprocess --input onnx_models/mnist_model.onnx --output onnx_models/mnist_model_processed.onnx
```
Do this after the ONNX conversion and before starting the quantization.
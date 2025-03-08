import torch
import openvino as ov
from utils import MODEL_DIR, test_loader, train_loader
from model import Net
import nncf

device = "cpu"

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_DIR / "original_model.pt"))

core = ov.Core()
example_input = next(iter(test_loader))[0]
ov_model = ov.convert_model(model, example_input=example_input)
ov.save_model(ov_model, MODEL_DIR / f"openvino_ir.xml")

def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

calibration_dataset = nncf.Dataset(train_loader, transform_fn)
quantized_model = nncf.quantize(ov_model, calibration_dataset)
model_int8 = ov.compile_model(quantized_model)
input_fp32 = next(iter(test_loader))[0][0:1]
res = model_int8(input_fp32)
ov.save_model(quantized_model, MODEL_DIR / f"quant_openvino_ir.xml")

def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

calibration_dataset = nncf.Dataset(train_loader, transform_fn)
quantized_model = nncf.quantize(ov_model, calibration_dataset)
model_int8 = ov.compile_model(quantized_model)
input_fp32 = next(iter(test_loader))[0][0:1]
res = model_int8(input_fp32)
ov.save_model(quantized_model, MODEL_DIR / f"quant_openvino_ir.xml")
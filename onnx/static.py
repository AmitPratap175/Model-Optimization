
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from utils import train_loader, MODEL_DIR
import onnxruntime

class QuantDR(CalibrationDataReader):
    def __init__(self, torch_data_loader, input_name):
        self.torch_data_loader = torch_data_loader
        self.input_name = input_name
        self.datasize = len(torch_data_loader)
        self.enum_data = iter(torch_data_loader)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_data_loader)

model_fp32 = MODEL_DIR / "mnist_model_processed.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_DIR / "mnist_model.onnx", providers=["CPUExecutionProvider"])

calibration_data = QuantDR(train_loader, ort_session.get_inputs()[0].name)
model__static_quant = MODEL_DIR / "mnist_model_static_quant.onnx"
static_quant_model = quantize_static(model_fp32, model__static_quant, calibration_data, weight_type=QuantType.QInt8)
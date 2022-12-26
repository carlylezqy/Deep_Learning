import os
import torch
from superresolution import init_torch_model
import onnx

def make_onnx(model, opt_path):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        torch.onnx.export(
            model, x, opt_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'])

def check_model(model_path):
    onnx_model = onnx.load(model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        return False
    else:
        return True

if __name__ == "__main__":
    spath = lambda path: os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

    model = init_torch_model(spath('srcnn.pth'))

    make_onnx(model, spath("srcnn.onnx"))
    print(check_model(spath("srcnn.onnx")))
    

    


import cv2
import numpy as np
import torch
import torch.onnx
from torch import nn
import os

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor, 
            mode='bicubic', 
            align_corners=False
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

class SuperResolutionVariableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = torch.nn.functional.interpolate(
            x, mode='bicubic', align_corners=False,
            scale_factor=upscale_factor,
        )
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model(path):
    torch_model = SuperResolutionNet()
    state_dict = torch.load(path)['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

def init_torch_model(path):
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load(path)['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

def main():
    pwd = os.path.abspath(os.path.dirname(__file__))
    spath = lambda path: os.path.join(pwd, path)

    model = init_torch_model(spath('srcnn.pth'))
    input_img = cv2.imread(spath('face.png')).astype(np.float32)

    # HWC to NCHW
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    input_img = torch.from_numpy(input_img)

    # Inference
    torch_output = model(input_img).detach().numpy()

    # NCHW to HWC
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

    # Show image
    cv2.imwrite(spath("face_torch.png"), torch_output)

if __name__ == "__main__":
    main()
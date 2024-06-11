import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ConvBNReLU(nn.Module):
  """Conv -> BatchNorm -> ReLU."""

  def __init__(self, input_shape, output_channels, kernel_shape, stride=(1,1), dilation=1, name=None):
    super(ConvBNReLU, self).__init__()
    self.name = name
    if stride == (1,1):
        padding1 = math.floor(((stride[0] - 1) * input_shape[1] - stride[0] + kernel_shape[0]) / 2.)
        padding2 = math.floor(((stride[1] - 1) * input_shape[2] - stride[1] + kernel_shape[1]) / 2.)
    else:
        padding1 = 1
        padding2 = 1
    self.conv = nn.Conv2d(in_channels=input_shape[0],
                          out_channels=output_channels, 
                          kernel_size=kernel_shape, 
                          stride=stride,
                          padding=(padding1, padding2),
                          dilation=dilation, 
                          bias=False)
    self.bn = nn.BatchNorm2d(num_features=output_channels)

  def forward(self, inputs):
    x = self.conv(inputs)
    x = self.bn(x)
    return F.relu(x)


class MultiBranchBlock(nn.Module):
    """Simple inception-style multi-branch block."""

    def __init__(self, input_shape, channels_1x1, channels_3x3, name=None):
        super().__init__()
        self.name = name

        self.conv1x1 = ConvBNReLU(input_shape=input_shape, output_channels=channels_1x1, kernel_shape=(1, 1), name="conv1x1").to(device)
        self.conv3x3 = ConvBNReLU(input_shape=input_shape, output_channels=channels_3x3, kernel_shape=(3, 3), name="conv3x3").to(device)

    def forward(self, inputs):
        conv1x1_out = self.conv1x1(inputs)
        conv3x3_out = self.conv3x3(inputs)
        output = torch.cat([conv1x1_out, conv3x3_out], dim=1)
        return output


class SmallInceptionStage(nn.Module):
  """Stage for SmallInception model."""

  def __init__(self, input_shape, mb_channels, downsample_channels, with_residual=False, name=None):
    super(SmallInceptionStage, self).__init__()
    self.name = name
    self._input_shape = input_shape
    self._mb_channels = mb_channels
    self._downsample_channels = downsample_channels
    self._with_residual = with_residual

    self._body = []
    for i, (ch1x1, ch3x3) in enumerate(mb_channels):
        self._body.append(MultiBranchBlock(input_shape=input_shape,
                                           channels_1x1=ch1x1,
                                           channels_3x3=ch3x3, 
                                           name=f"block{i+1}"))
        input_shape = (ch1x1+ch3x3, input_shape[1], input_shape[2])
    if downsample_channels > 0:
        self._downsample = ConvBNReLU(input_shape=input_shape,
                                      output_channels=downsample_channels,
                                      kernel_shape=(3, 3),
                                      stride=(2, 2),
                                      name="downsample")
    else:
        self._downsample = None

  def forward(self, inputs):
    x = inputs
    for block in self._body:
        x = block(x)
    if self._with_residual:
        x += inputs

    if self._downsample:
        x = self._downsample(x)
    return x



class Inception(nn.Module):
    def __init__(self, num_classes=10, with_residual=False, large_inputs=False, name=None):
        super().__init__()
        self.name = name
        self._num_classes = num_classes

        self._large_inputs = large_inputs
        if large_inputs:
            self.conv1 = ConvBNReLU(input_shape=(3, 32, 32),
                                    output_channels=96,
                                    kernel_shape=(7, 7),
                                    stride=2,
                                    name="conv1").to(device)
        else:
            self.conv1 = ConvBNReLU(input_shape=(3, 32, 32),
                                    output_channels=96,
                                    kernel_shape=(3, 3),
                                    name="conv1").to(device)
        self.stage1 = SmallInceptionStage(input_shape=(96, 32, 32),
                                          mb_channels=[(32, 32), (32, 48)], 
                                          downsample_channels=160,
                                          with_residual=with_residual).to(device)
        self.stage2 = SmallInceptionStage(input_shape=(160, 16, 16),
                                          mb_channels=[(112, 48), (96, 64), (80, 80), (48, 96)],
                                          downsample_channels=240,
                                          with_residual=with_residual).to(device)
        self.stage3 = SmallInceptionStage(input_shape=(240, 8, 8),
                                          mb_channels=[(176, 160), (176, 160)],
                                          downsample_channels=0,
                                          with_residual=with_residual).to(device)
        self._pred = nn.Linear(in_features=336,
                               out_features=num_classes).to(device)

    def compute_repr(self, inputs):
        x = inputs
        if self._large_inputs:
            x = self.conv1(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
        else:
            x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # global pooling
        x = torch.amax(x, dim=[2, 3])
        return x

    def forward(self, inputs):
        logits = self._pred(self.compute_repr(inputs))
        return logits
        


if __name__=='__main__':
    checkpoint_root = "checkpoints/inception"
    checkpoint_name = "torch.pt"
    save_path = os.path.join(checkpoint_root, checkpoint_name)
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root, exist_ok=True)
    
    model = Inception()
    torch.save(model, save_path)
    summary(model, (3, 32, 32))

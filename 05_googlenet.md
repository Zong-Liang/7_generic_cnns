# googlenet

![inception](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202401111306114.svg)

![inception-full](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202401111311024.svg)

```python
import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
print(net)

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t", X.shape)
```

```python
Sequential(
  (0): Sequential(
    (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)     
  )
  (1): Sequential(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)     
  )
  (2): Sequential(
    (0): Inception(
      (p1_1): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))        
      (p3_1): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (p4_2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Inception(
      (p1_1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))       
      (p3_1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (3): Sequential(
    (0): Inception(
      (p1_1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Inception(
      (p1_1): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Inception(
      (p1_1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Inception(
      (p1_1): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (4): Inception(
      (p1_1): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (4): Sequential(
    (0): Inception(
      (p1_1): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Inception(
      (p1_1): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
      (p2_1): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
      (p2_2): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (p3_1): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
      (p3_2): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (p4_1): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)        
      (p4_2): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): AdaptiveAvgPool2d(output_size=(1, 1))
    (3): Flatten(start_dim=1, end_dim=-1)
  )
  (5): Linear(in_features=1024, out_features=10, bias=True)
)
```

```python
Sequential output shape: torch.Size([1, 64, 24, 24])
Sequential output shape: torch.Size([1, 192, 12, 12])
Sequential output shape: torch.Size([1, 480, 6, 6])
Sequential output shape: torch.Size([1, 832, 3, 3])
Sequential output shape: torch.Size([1, 1024])
Linear output shape:     torch.Size([1, 10])
```


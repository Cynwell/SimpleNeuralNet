import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        return out

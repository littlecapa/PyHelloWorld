from torch import nn
from nn_hash import nn_Hash

class HW_Machine(nn_Hash):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.reshape(1)
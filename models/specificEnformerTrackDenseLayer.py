from torch import nn
import torch.nn.functional as f

'''
Basic dense layer that is configured to have input layer dimensions when only lung and blood specific Enformer tracks 
are used as features (73 features)
'''
class BasicDenseLayer(nn.Module):
    def __init__(self):
        super(BasicDenseLayer, self).__init__()
        self.fc1 = nn.Linear(73, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
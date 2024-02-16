from torch import nn
import torch.nn.functional as f

#Feed forward neural network with 2 hidden layers
class BasicDenseLayer(nn.Module):
    def __init__(self):
        super(BasicDenseLayer, self).__init__()
        self.fc1 = nn.Linear(5313, 2500)
        self.fc2 = nn.Linear(2500, 1000)
        self.fc3 = nn.Linear(1000, 200)    
        self.fc4 = nn.Linear(200, 2)

    '''
    Inputs - 
    x -> Enformer predictions read from Enformer output H5PY files. Shape: (batch_size * 5313) where 5313 comes from the 
        number of Enformer output tracks for a single sample. 

    Output - 
    Predictions by the model for all samples in this batch (batch_size * 2) (col 1 is a measure of the likelihood of 
    the sample being donor-derived and col 2 is the likelihood for being recipient-derived) 
    '''
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x
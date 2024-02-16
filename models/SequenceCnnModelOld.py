import torch
from torch import nn
import torch.nn.functional as f

class SequenceCnnModelOld(nn.Module):
    def __init__(self, dropoutProbability):
        super(SequenceCnnModelOld, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, (15, 4))
        self.pool1 = nn.MaxPool2d((4, 1))
        self.conv2 = nn.Conv2d(20, 40, (15, 1))
        self.pool2 = nn.MaxPool2d((13, 1))
        self.fc1 = nn.Linear(200, 100) #Not ideal, but remember to increase this by 1, if padding is set to true
        self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(dropoutProbability)

    def forward(self, x, sequence_length, addLengthAsAFeature):
        #Pass Sequence through a CNN followed by max pooling layer to identify motifs
        x = f.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool2(x)

        #Flatten outputs of the channels into additional feaatures
        x = torch.flatten(x, start_dim=1)

        #Add length of the sequence as an extra feature
        if(addLengthAsAFeature == True):
            x = torch.cat((x, sequence_length),-1)
        
        #Pass the CNN outputs through a dense layer for the final classification
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
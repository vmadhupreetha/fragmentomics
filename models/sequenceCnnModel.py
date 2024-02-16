import torch
from torch import nn
import torch.nn.functional as f

'''
Deep Learning model 2: CNN mdel that extracts sequence motifs using Convolutional filters and passes the resulting motifs
through feed forward neural network layers for classification. 

2 layer CNN model (each layer followed by max pooling). 1 hidden layer feed forward neural network for classification. 
'''
class SequenceCnnModel(nn.Module):
    '''
    Input -
    dropoutProbability - value between 0 and 1. In this model, dropout layers are added after every layer (refer to
        forward method). This is a method to reduce overfitting by dropping the output of some neurons after every layer.
        Dropout probability indicates the proportion of neurons whose outputs will be dropped.
    '''
    def __init__(self, dropoutProbability):
        super(SequenceCnnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (15, 4))
        self.pool1 = nn.MaxPool2d((4, 1))
        self.conv2 = nn.Conv2d(40, 80, (15, 1))
        self.pool2 = nn.MaxPool2d((15, 1))
        self.fc1 = nn.Linear(401, 200) #Not ideal, but remember to increase this by 1, if padding is set to true
        self.fc2 = nn.Linear(200, 2)
        self.dropout = nn.Dropout(dropoutProbability)

    '''
    Inputs -
    1. x -> One hot encoded cfDNA fragment sequence. Shape: (batch_size * 370 * 4) where 370 is the number of bases in the sequence
        and 4 comes from one-hot encoding. 
    2. sequence_length -> original length of the cfDNA fragment (this is different from 370; 370 is the extended length)
    3. useLengthAsFeature -> Boolean value, if it is true then the last feed forward layer will use sequence_length 
        in addition to extracted motifs for the classification. 

    Output - 
    Predictions by the model for all samples in this batch (batch_size * 2) (col 1 is a measure of the likelihood of 
    the sample being donor-derived and col 2 is the likelihood for being recipient-derived) 
    '''
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
        x = f.relu(self.fc2(x))
        return x
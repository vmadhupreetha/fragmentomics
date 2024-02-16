import torch
from torch import nn
import torch.nn.functional as f

'''
Deep Learning model 3: A CNN model that extracts sequence motifs from raw cfDNA sequence and combines it with Enformer 
predictions for the sequence for making the final classification. 
2 convolutional layers (followed by max pooling layers), followed by a 1 hidden layer feed forward neural network 
'''
class combinedSequenceEnformerModel(nn.Module):
    '''
    Input -
    dropoutProbability - value between 0 and 1. In this model, dropout layers are added after every layer (refer to
        forward method). This is a method to reduce overfitting by dropping the output of some neurons after every layer.
        Dropout probability indicates the proportion of neurons whose outputs will be dropped.
    '''
    def __init__(self, dropoutProbability):
        super(combinedSequenceEnformerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (15, 4))
        self.pool1 = nn.MaxPool2d((4, 1))
        self.conv2 = nn.Conv2d(40, 80, (15, 1))
        self.pool2 = nn.MaxPool2d((15, 1))
        self.fc1 = nn.Linear(5714, 2000) 
        self.fc2 = nn.Linear(2000, 200)
        self.fc3 = nn.Linear(200, 2)
        self.dropout = nn.Dropout(dropoutProbability)

    '''
    Inputs -
    1. x -> One hot encoded cfDNA fragment sequence. Shape: (batch_size * 370 * 4) where 370 is the number of bases in the sequence
        and 4 comes from one-hot encoding. 
    2. y -> Enformer prediction read from Enformer output H5PY files. Shape: (batch_size * 5313) where 5313 comes from the 
        number of Enformer output tracks for a single sample. 
    3. sequence_length -> original length of the cfDNA fragment (this is different from 370; 370 is the extended length)
    4. useLengthAsFeature -> Boolean value, if it is true then the last feed forward layer will use sequence_length 
        in addition to extracted motifs and enformer predictions for the classification. 
        
    Output - 
    Predictions by the model for all samples in this batch (batch_size * 2) (col 1 is a measure of the likelihood of 
    the sample being donor-derived and col 2 is the likelihood for being recipient-derived) 
    '''
    def forward(self, x, y, sequence_length, useLengthAsFeature):
        #Pass sequence through CNNs followed by max pooling layers to capture motifs
        x = f.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool2(x)

        #Flatten outputs of the channels into additional feaatures
        x = torch.flatten(x, start_dim=1) 

        #Combine CNN output from sequence and enformer outputs
        y.requires_grad=True
        combined_sample = torch.cat((x, y), 1)

        #add length feature to the combined outputs
        if(useLengthAsFeature == True): 
            combined_sample = torch.cat((combined_sample, sequence_length), 1)

        #Normalize each feature of combined sample so the 3 different feature types have comparable scales. 
        combined_sample = f.normalize(combined_sample, dim=0, p = 1)

        combined_sample = f.relu(self.fc1(combined_sample))
        combined_sample = self.dropout(combined_sample)
        combined_sample = f.relu(self.fc2(combined_sample))
        combined_sample = self.dropout(combined_sample)
        output = self.fc3(combined_sample)
        return output
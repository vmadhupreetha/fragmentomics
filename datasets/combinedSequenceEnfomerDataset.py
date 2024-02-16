'''
This dataset reads the H5PY file containing one hot encoded sequences and the H5PY file containing enformer outputs in parallel
The output from this dataset is used for training the combined sequence and enformer output model.

The dataset returns a tuple of the following 4 values
torch_sequence_output -> Tensor of size (batchsize, 330, 4). It is the one hot encoded sequence with 330 bases.
torch_enformer_output -> Tensor of size (batchSize, 5313) if averageEnformerOutputBins is True. Otherwise it is (batchSize, 10626). It is the Enformer output
og_sequence_length ->  Tensor of size (batchSize, 1). Contains the length of the one hot encoded sequence (330 is the length after padding)
class_labels -> Tensor of size (batchSize, 1). Has the class labels for each sample (0 = donor class and 1 = recipient class)
'''

from torch.utils.data import Dataset
import utils
import config 
import h5py
import numpy as np
import torch

import importlib
importlib.reload(utils)
importlib.reload(config)

arguments = {}

#File paths 
arguments["trainingEnformerOutputStoreFile"] = config.filePaths.get("trainingEnformerOutputStoreFile")
arguments["validationEnformerOutputStoreFile"] = config.filePaths.get("validationEnformerOutputStoreFile")
arguments["testEnformerOutputStoreFile"] = config.filePaths.get("testEnformerOutputStoreFile")
arguments["trainingSequenceFile"] = config.filePaths.get("trainingEncodedSequenceFilePath")
arguments["validationSequenceFile"] = config.filePaths.get("validationEncodedSequenceFilePath")
arguments["testSequenceFile"] = config.filePaths.get("testEncodedSequenceFilePath")

#Enformer related configs 
arguments["trainingStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesTraining")
arguments["trainingEndIndex"] = config.modelGeneralConfigs.get("endIndexEnformerSamplesTraining")
arguments["validationStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesValidation")
arguments["validationEndIndex"] = config.modelGeneralConfigs.get("endIndexEnformerSamplesValidation")
arguments["averageEnformerOutputBins"] = config.modelGeneralConfigs.get("averageEnformerOutputBins")

#general configs
arguments["runWithControls"] = config.modelGeneralConfigs.get("runWithControls")
arguments["interchangeLabels"] = config.modelGeneralConfigs.get("interchangeLabels")
arguments["useClassWeights"] = config.modelGeneralConfigs.get("useClassWeights")

#Dataset names 
#Labels datasets 
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

#One hot encoded sequence datasets 
arguments["trainingEncodedSequenceDatasetName"] = config.datasetNames.get("trainingEncodedSequence")
arguments["validationEncodedSequenceDatasetName"] = config.datasetNames.get("validationEncodedSequence")
arguments["testEncodedSequenceDatasetName"] = config.datasetNames.get("testEncodedSequence")
arguments["trainingSequenceLengthDatasetName"] = config.datasetNames.get("trainingSequenceLength")
arguments["validationSequenceLengthDatasetName"] = config.datasetNames.get("validationSequenceLength")
arguments["testSequenceLengthDatasetName"] = config.datasetNames.get("testSequenceLength")

#Enformer output datasets
arguments["trainingEnformerOutputDatasetName"] = config.datasetNames.get("trainingEnformerOutput")
arguments["validationEnformerOutputDatasetName"] = config.datasetNames.get("validationEnformerOutput")
arguments["testEnformerOutputDatasetName"] = config.datasetNames.get("testEnformerOutput")

print(f"arguments in file combinedSequenceEnformerDataset are {arguments}")

class combinedSequenceEnfomerDataset(Dataset):
    '''
    sampleType - training, validation or test. Depending on the sampleType, different H5PY datasets are read.
    '''
    def __init__(self, sampleType):
        self.sampleType = sampleType
        self.enformerOutputDatasetName = arguments[sampleType + "EnformerOutputDatasetName"]
        self.labelsDatasetName = arguments[sampleType + "LabelsDatasetName"]
        self.encodedSequenceDatasetName = arguments[sampleType + "EncodedSequenceDatasetName"]
        self.sequenceLengthDatasetName = arguments[sampleType + "SequenceLengthDatasetName"]

        self.enformerOutputFilePath = arguments[sampleType + "EnformerOutputStoreFile"]
        self.encodedSequenceFilePath = arguments[sampleType + "SequenceFile"]
        self.startIndex = arguments[sampleType + "StartIndex"]
        self.endIndex = arguments[sampleType + "EndIndex"]

    '''
    This method is called when iterating over the dataset, it fetches a batch of random samples, whose indices are specified
    by indices list.
    
    Input - 
    Indices is a list of random indices in the range [0: length of the dataset] (length comes from __len__ function). 
    Two HYPY files are read simultaneouly - the encodedSequenceFile (one hot encoded cfDNA fragment sequences)
        and the enformerOutputFile (enformer predictions for cfDNA sequence). The files were created such that 
        the index for a given sample is same in both the files. 
    '''
    def __getitem__(self, indices):
        try:
            #Read encoded sequence from H5PY file 
            with h5py.File(self.encodedSequenceFilePath, 'r') as f:
                sequenceOutput = f[self.encodedSequenceDatasetName][indices]
                og_sequence_length = f[self.sequenceLengthDatasetName][indices]
                torch_sequence_output = torch.tensor(np.float32(sequenceOutput))

            #Read Enformer output from H5PY file 
            with h5py.File(self.enformerOutputFilePath, 'r') as f:
                enformerOutput = f[self.enformerOutputDatasetName][indices]
                torch_enformer_output = torch.tensor(np.float32(enformerOutput))
                labels = f[self.labelsDatasetName][indices]

            #Average outputs from the 2 enformer bins 
            torch_enformer_output = self.averageEnformerOutputBins(torch_enformer_output)

            # specific_indices = [105, 106, 111, 138, 167, 200, 229, 231, 245, 272, 279, 295, 314, 327, 387, 388, 390, 396, 400, 406, 407, 409, 419, 437, 461, 464, 487, 492, 512, 569, 579, 589, 590, 591, 603, 627, 632, 645, 649, 655, 659, 668, 41, 131, 392, 508, 517, 188, 351, 366, 488, 489, 214, 233, 346, 369, 410, 454, 588, 598, 634, 61, 211, 214, 233, 333, 346, 369, 410, 454, 588, 598, 634]
            # torch_enformer_output = torch_enformer_output[:, specific_indices]

            #For old coordinate files, the positives were incorrectly labelled as 0 and the negatives as 1.
            if(arguments["interchangeLabels"]):
                labels = self.interchangeLabels(labels)

            return torch_sequence_output, torch_enformer_output, og_sequence_length, labels
        except Exception as e : 
            print(f"Caught an exception for indices : {indices}. The exception is {e}")
            raise

    '''
    Length of dataset - determines range of indices. typically the length of the enformerOutput file (endIndex is "all") 
    But in some cases, only a portion of the EnformerOutput store file has correct data 
    (when the process of predicting output and storing into file gets interrupted, the indices that were not processed
    are populated with 0s). In this case, endIndex is the last index in the EnformerOutputFile that has correct values. 
    '''
    def __len__(self):
        with h5py.File(self.enformerOutputFilePath, 'r') as f:
            if self.endIndex == "all":
                labels = f[self.labelsDatasetName][self.startIndex:]
            else: 
                labels = f[self.labelsDatasetName][self.startIndex:self.endIndex]

        return len(labels)

    """
    Returns the class weights to be used on this dataset, to counter class imbalance. 
    Outputs an array of 2 elements - the 1st element is the weight for positives and 2nd is the weight for the negatives
    At present, since the training samples have equal number of donors and recipients, this function always returns [1,1]
    """
    def getClassWeights(self):
        return [1, 1]

    """
    Given a numpy labels array of 1's and 0's, it interchanges the 1's with the 0's and nice versa
    Output -> 
    """
    def interchangeLabels(self,labels):
        positive_indices = np.where(labels == 0)
        negative_indices = np.where(labels == 1)
        labels[positive_indices] = 1
        labels[negative_indices] = 0
        return labels
    

    #It is easier to combine the bins before enformer output file is generated. This function is there in case 
    #the file is already created and outputs need to be combined while reading. 
    def averageEnformerOutputBins(self, samples):
        _, numFeatures = samples.shape
        first_bin_end_index = int(numFeatures/2)
        bin_averaged_output = torch.empty((1,first_bin_end_index))
        for sample in samples:
            first_bin_outputs = sample[0:first_bin_end_index].reshape(1, first_bin_end_index)
            second_bin_outputs = sample[first_bin_end_index: numFeatures].reshape(1, first_bin_end_index)
            average = (first_bin_outputs + second_bin_outputs)/2
            bin_averaged_output = torch.cat((bin_averaged_output, average), 0)
            
        return bin_averaged_output[1:, :]

'''
This is a Pytorch dataset class. Upon iterating through objects of this class, a random sample from a random coordinateFile
in the given coordStoreDirectory is processed. A "sample" in a coordinateFile is the coordinates of a single cfDNA fragment
defined by the chromosome number, start and end coordinates. For a single sample, the DNA sequence corresponding to the
coordinates and some additional metadata are returned.

Upon iterating through the dataset, the __get_item() method (below) is called. Refer to this method for more details on the
processing and the objects returned.

The dataset is specific to the sampleType - training/validation/test. Samples from different H5PY datasets are processed
depending on the sampleType.
'''

from torch.utils.data import Dataset
import os
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
arguments["coordStoreDirectory"] = config.filePaths.get("coordStoreDirectory")
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")

#General configs related to model intput data
arguments["modelInputSequenceSize"] = config.modelHyperParameters.get("modelInputSequenceSize")
arguments["runWithControls"] = config.modelGeneralConfigs.get("runWithControls")
arguments["usePaddingForCnn"] = config.modelGeneralConfigs.get("usePaddingForCnn")
arguments["interchangeLabels"] = config.modelGeneralConfigs.get("interchangeLabels")

#DatasetNames
arguments["trainingCoordsDatasetName"] = config.datasetNames.get("trainingCoords")
arguments["validationCoordsDatasetName"] = config.datasetNames.get("validationCoords")
arguments["testCoordsDatasetName"] = config.datasetNames.get("testCoords")
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

print(f"arguments in file sequenceDataset are {arguments}")

class SequenceDataset(Dataset):
    #set some variables when the dataset is initialized.
    def __init__(self, sampleType):
        self.sampleType = sampleType
        self.coordDatasetName = arguments[sampleType + "CoordsDatasetName"]
        self.labelsDatasetName = arguments[sampleType + "LabelsDatasetName"]
        self.startIndexList, self.fileNamesList = utils.createFileNameIndexList(arguments["coordStoreDirectory"], self.coordDatasetName)

    '''
    This method is called internally upon iterating through this dataset. A random index is passed everytime (without repetition)
    Does the following
    1. Fetch the sample for the index (The random index is for all the samples from all coordinate files in the directory
        So from the random index, we need to get the coordinate file and the index within the file)
    2. Call functions to get one hot encoded sequence from coordinates. 
    
    Output
    1. The one hot encoded sequence for coordinate 
    2. Label for the sample (0 if recipient-derived and 1 if donor-derived) 
    3. bins - Enformer output bins which correspond the original sequence (refer to getOneHotEncodedSequenceFromCoordinates
        for more backstory about why these bins are needed)
    5. filePath - the path of the coordinateFile from which the sample for this index is returned
    6. indexWithinFile - The index within this coordinatefile
    7. og_sequence_length - length of the sequence specified by the coordinates (Note that before one hot encoding,
        the coordinates are extended to reach the enformer or CNN input length. So length of the returned sequence 
        is not the same as the length of the sequence specified by the coordinates) 
    '''
    def __getitem__(self, index):
        filePosition = utils.getFilePositionFromIndex(self.startIndexList, index)
        filename = self.fileNamesList[filePosition]            
        indexWithinFile = index - self.startIndexList[filePosition]
        numLinesFile = utils.getNumberLinesInFile(arguments["coordStoreDirectory"], filename, self.coordDatasetName)
        if(indexWithinFile >  numLinesFile + 5):
            print(f"For index: {index}, indexWithinFile: {indexWithinFile} far exceeds numLinesFile: {numLinesFile} for file: {filename}", flush=True)
        
        filepath = os.path.join(arguments["coordStoreDirectory"], filename)
        with h5py.File(filepath, 'r') as f:
            coord = f[self.coordDatasetName][indexWithinFile]

            #Each sample should have only one label, it should be a single value instead of a numpy 1D array.The [0] is to make it a single value instead of a numpy array.
            # label = f['trainingLabels'][index][0]
            label = f[self.labelsDatasetName][:][indexWithinFile]
            if(arguments["interchangeLabels"] == True):
                labels = self.interchangeLabels(label)

            sequenceOutputLength = arguments["modelInputSequenceSize"]
            encoded_input_sequence, bins, og_sequence_length = utils.getOneHotEncodedSequenceFromCoordinates(coord, arguments["refGenomePath"],
                                                                                         sequenceOutputLength, arguments["usePaddingForCnn"])
            
            #For some cases, the coordinates look fine, but the sequence fetched from the fasta file has size 0. 
            #If we pass such samples to enformer for predictions, we get Einops error, due to dimension mismatch.
            expected_sequence_length = 196607 if sequenceOutputLength == "enformer" else sequenceOutputLength
            assert encoded_input_sequence.shape == (expected_sequence_length, 4), f"One of the samples did not have the right dimensions({(expected_sequence_length, 4)}). The sample index is {index}, shape is {encoded_input_sequence.shape}, filename is {filename} and index within the file is {indexWithinFile}"
            
            if(arguments["runWithControls"]):
                encoded_input_sequence = self.addPositiveAndNegativeControls(encoded_input_sequence, label[0])
        
        return encoded_input_sequence, label, bins, filepath, indexWithinFile, og_sequence_length 

    '''
    Inbuilt method in dataset which returns the number of samples in the dataset. This function will be called to determine
    the range of indices that will be passed to the __get_item() method. 
    '''
    def __len__(self):
        #Total number of samples is the startIndexList of the last file + number of samples in the last file. 
        lastStartIndex = self.startIndexList[-1]
        lastFileName = self.fileNamesList[-1]
        numSamplesLastFile = utils.getNumberLinesInFile(arguments["coordStoreDirectory"], lastFileName, self.coordDatasetName)
        totalNumSamples = lastStartIndex + numSamplesLastFile
        return totalNumSamples
    
    #TODO implement proper training class weights
    '''
    Original intent was to use this for class balancing - to prevent the model from bias towards the majority class. 
    This would return a tuple of the weights to be used for determining loss function during training. These weights 
    would be the inverse of the proportion of samples belonging to respective classes, such that the minority class 
    receives a higher weight. 
    
    For this project, downsampling was done to balance the classes so class weights was not used in the end. 
    '''
    def getClassWeights(self):
        return [1, 1]

    '''
    For some coordinate files, the labels were accidentally interchanged (0 donor and 1 recipient). This method is to 
    change them back while working with these coordinate files. 
    '''
    def interchangeLabels(self,labels):
        positive_indices = np.where(labels == 0)
        negative_indices = np.where(labels == 1)
        labels[positive_indices] = 1
        labels[negative_indices] = 0
        return labels

    '''
    Add A's at the end if sample is recipient-derived and T's at the end if sample is donor-derived (The one hot encoded 
    versions of As and Ts are added). This is a test to investigate model vs data issues by clearly adding a signal
    separating the two classes.  
    '''
    def addPositiveAndNegativeControls(self,sequence, class_label):
        num_controls = 4
        if class_label == 0:
            single_control = np.array([1, 0, 0, 0])
        else:
            single_control = np.array([0, 0, 0, 1])

        controls = np.tile(single_control, (num_controls, 1))
        controls = torch.tensor(np.float32(controls))

        #Replace the last num_controls bases in the sequence with the one hot encoded controls. 
        nfeats, _ = sequence.shape
        nfeats_without_control = nfeats - num_controls
        sequence_without_controls = sequence[0:nfeats_without_control, :]
        sequence = torch.cat((sequence_without_controls, controls), 0)
        return sequence
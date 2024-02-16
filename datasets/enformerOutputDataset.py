'''
This is a Pytorch dataset class. Upon iterating through objects of this class, a random Enformer prediction
sample from the configured EnformerOutput file is processed and returned.

Upon iterating through the dataset, the __get_item() method (below) is called. Refer to this method for more details on the
processing and the objects returned.
'''
import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import importlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import math

import h5py
import config 
import utils

from sklearn.utils.class_weight import compute_class_weight

importlib.reload(utils)
importlib.reload(config)

#Set arguments from config file.
arguments = {}
#File paths
arguments["trainingEnformerOutputStoreFile"] = config.filePaths.get("trainingEnformerOutputStoreFile")
arguments["validationEnformerOutputStoreFile"] = config.filePaths.get("validationEnformerOutputStoreFile")
arguments["testEnformerOutputStoreFile"] = config.filePaths.get("testEnformerOutputStoreFile")

#Enformer output file related configs. 
arguments["trainingStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesTraining")
arguments["trainingEndIndex"] = config.modelGeneralConfigs.get("endIndexEnformerSamplesTraining")
arguments["validationStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesValidation")
arguments["validationEndIndex"] = config.modelGeneralConfigs.get("endIndexEnformerSamplesValidation")

#General configs
arguments["useClassWeights"] = config.modelGeneralConfigs.get("useClassWeights")
arguments["interchangeLabels"] = config.modelGeneralConfigs.get("interchangeLabels")
arguments["averageEnformerOutputBins"] = config.modelGeneralConfigs.get("averageEnformerOutputBins")

#DatasetNames
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")
arguments["trainingEnformerOutputDatasetName"] = config.datasetNames.get("trainingEnformerOutput")
arguments["validationEnformerOutputDatasetName"] = config.datasetNames.get("validationEnformerOutput")
arguments["testEnformerOutputDatasetName"] = config.datasetNames.get("testEnformerOutput")

#Configurations for running with controls 
arguments["runWithControls"] = config.modelGeneralConfigs.get("runWithControls")
arguments["trainingNumSimulatedSamples"] = config.modelGeneralConfigs.get("numSimulatedTrainingSamples")
arguments["validationNumSimulatedSamples"] = config.modelGeneralConfigs.get("numSimulatedValidationSamples")

# !!!!!!! All the test configs (Comment this out when using this for actual runs) !!!!!!
# arguments["trainingEnformerOutputStoreFile"] = config.testFilePaths.get("trainingEnformerOutputStoreFile")
# arguments["validationEnformerOutputStoreFile"] = config.testFilePaths.get("validationEnformerOutputStoreFile")
# arguments["runWithControls"] = config.testConfigs.get("runWithControls")
# arguments["trainingNumSimulatedSamples"] = config.testConfigs.get("numSimulatedTrainingSamples")
# arguments["validationNumSimulatedSamples"] = config.testConfigs.get("numSimulatedValidationSamples")


class EnformerOutputDataset(Dataset):
    def __init__(self, sampleType, normalizeFeatures, trackAverages = False, percent_features = False, percent_samples = False):
        self.sampleType = sampleType
        self.enformerOutputDatasetName = arguments[sampleType + "EnformerOutputDatasetName"]
        self.labelsDatasetName = arguments[sampleType + "LabelsDatasetName"]
        
        self.enformerOutputFilePath = arguments[sampleType + "EnformerOutputStoreFile"]
        self.startIndex = arguments[sampleType + "StartIndex"]
        self.endIndex = arguments[sampleType + "EndIndex"]
        self.normalizeFeatures = normalizeFeatures 
        self.trackAverages = trackAverages

        self.percent_features = percent_features
        self.percent_samples = percent_samples
    
    """
    Return Enformer predictions and labels from the EnformerOutput file for a list of random indices 
    Inputs : 
    Indices will be an array of indices. The size of the array is equal to the batch size. 
    At a time, the entire batch will be loaded. 
    
    The indexes fetched by dataloader iteration are not in order, because shuffling is set to true. This will not cause a mismatch
    between the enformer output and the label. Because enformer output and label are fetched for the same index, so they will still
    correspond to each other. 
    
    Output - 
    1. encoded_enformer_output - Pytorch tensor of enformer prediction (could be size batch_size * 5313 or batch_size * 10626
        depending on whether averageEnformerOutputBins is true) 
    2. labels - Pytorch tensor of labels of size (batch_size * 1)
    """
    def __getitem__(self, indices):
        '''
        If this argument is set to true, instead of true data from enformerOutputStore file, simulated data with
        artificially augmented signals will be returned.
        '''
        if(arguments["runWithControls"] == True):
            simulated_data, labels = self.getSimulatedData(self.trackAverages, len(indices), self.percent_features, self.percent_samples)
            simulated_data = torch.tensor(np.float32(simulated_data))
            return simulated_data, labels
        
        with h5py.File(self.enformerOutputFilePath, 'r') as f:
            enformer_output = f[self.enformerOutputDatasetName][indices]
            labels = f[self.labelsDatasetName][indices]

        encoded_enformer_output = torch.tensor(np.float32(enformer_output))
            
        #Enformer output file has the predictions from 2 bins. Average them to reduce feature size.
        if(arguments['averageEnformerOutputBins']):
            encoded_enformer_output = self.averageEnformerOutputBins(encoded_enformer_output)
        
        # specific_indices = [105, 106, 111, 138, 167, 200, 229, 231, 245, 272, 279, 295, 314, 327, 387, 388, 390, 396, 400, 406, 407, 409, 419, 437, 461, 464, 487, 492, 512, 569, 579, 589, 590, 591, 603, 627, 632, 645, 649, 655, 659, 668, 41, 131, 392, 508, 517, 188, 351, 366, 488, 489, 214, 233, 346, 369, 410, 454, 588, 598, 634, 61, 211, 214, 233, 333, 346, 369, 410, 454, 588, 598, 634]
        # encoded_enformer_output = encoded_enformer_output[:, specific_indices]

        #Each Enformer track could have different ranges of values. Take row wise z-score, so values of one track do not dominate.
        if(self.normalizeFeatures):
            encoded_enformer_output = self.normalizeFeatures(encoded_enformer_output, self.enformerOutputFilePath, self.enformerOutputDatasetName)
            print(f"Normalized Enformer output shape is {encoded_enformer_output.shape}")
            
        #For old coordinate files, the positives were incorrectly labelled as 0 and the negatives as 1.
        if(arguments["interchangeLabels"]):
            labels = self.interchangeLabels(labels)
            
        return encoded_enformer_output, labels

    '''
    Length of dataset - determines range of indices. typically the length of the enformerOutput file (endIndex is "all") 
    But in some cases, only a portion of the EnformerOutput store file has correct data 
    (when the process of predicting output and storing into file gets interrupted, the indices that were not processed
    are populated with 0s). In this case, endIndex is the last index in the EnformerOutputFile that has correct values. 
    '''
    def __len__(self):
        if(arguments["runWithControls"] == True):
            numSamples = arguments[self.sampleType + "NumSimulatedSamples"]
            return numSamples
        
        #If the model is not being run with simulated data, then read the enformer output files to get the number of samples. 
        with h5py.File(self.enformerOutputFilePath, 'r') as f:
            if self.endIndex == "all":
                labels = f[self.labelsDatasetName][self.startIndex:]
            else: 
                labels = f[self.labelsDatasetName][self.startIndex:self.endIndex]

            return len(labels)
    
    """
    Depending on the distribution of positives and negatives in the data, get the class weights such that the 
    class numbers can be made equal. 
    """
    def getClassWeights(self):
        if(arguments["useClassWeights"] != True):
            return [1,1]
        
        """
        We are only using the 1st 10,000 samples to compute the class weights, assuming they are reflective
        of the training vs validation distribution of the whole sample. 
        """
        with h5py.File(arguments["trainingEnformerOutputStoreFile"], 'r') as f:
            training_labels = f["trainingLabels"][0:10000].flatten().tolist()
        
        with h5py.File(arguments["validationEnformerOutputStoreFile"], 'r') as f:
            validation_labels = f["validationLabels"][0:10000].flatten().tolist()
        
        all_labels = training_labels + validation_labels
        class_weights = compute_class_weight(class_weight = "balanced", classes = [0, 1], y = all_labels)
        return class_weights

    """
    Take a 2D tensor where rows are samples and columns are features. Normalise each feature by taking z scores of the feature 
    across all samples. 
    For calculating mean and std, use only the first 10000 samples in the enformerOutputStore file for easy computation. 
    Inputs - 
    1. enformer_output - Enformer output to be normalized 
    2. enformerOutputFilePath - 
    3. enformerOutputDataset - The dataset name (depending on sampleType) in the enformerOutputStoreFile (for getting mean)
    """
    def normalizeFeatures(enformer_output, enformerOutputFilePath, enformerOutputDataset):
        #Read the 1st 10000 values from the file to get mean and std. If we take mean and std only within the batch the variation is too low. 
        with h5py.File(enformerOutputFilePath, 'r') as f:
            enformerOutputForMean = f[enformerOutputDataset][0:10000]
        mean = torch.mean(enformerOutputForMean, axis = 1) #Mean for each feature
        std = torch.std(enformerOutputForMean, axis = 1)

        print(f"Shape of mean is {mean.shape}, some values : {mean[0][0:15]}")
        print(f"Shape of std : {std.shape}, some values : {std[0][0:15]}")

        t = enformer_output.reshape(1, enformer_output.shape[0], enformer_output.shape[1]) #Get the tensor to the format of (1, nsamples, nfeatures)
        t = tf.normalize(t, mean, std)
        t = t.reshape(t.shape[1], t.shape[2]) #Reshape the tensor to original dimension

        return t 

    '''
    Some files have interchanged labels. So after reading the labels, replace 0's with 1's and 1s with 0s. 
    '''
    def interchangeLabels(self,labels):
        positive_indices = np.where(labels == 0)
        negative_indices = np.where(labels == 1)
        labels[positive_indices] = 1
        labels[negative_indices] = 0
        return labels

    '''
    It is easier to combine the bins before enformer output file is generated. This function is there in case 
    the file is already created and outputs need to be combined while reading. 
    '''
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

    """
    This function first generates random values of simulated Enformer tracks based on the track's minimum and maximum
    Then, signal is artifically addded into the data for a specific number of features or samples, as configured. 
    
    Inputs - 
    1. track_ranges - A map with "min" and "max" as keys. 
        The values of "min" and "max" are lists of minimums and maximums for each track 
    2. num_samples - Number of simulated data samples
    3. percent_features - percentage of features to be augmented with signals
    4. percent_samples - percentage of samples to be augmented with signals. 
    
    Output - 
    1. simulated_samples - numpy array of size (num_sammples * num_tracks) of simulated values. 
    2. labels - numpy array of size (num_samples) of 1s and 0s. Labels are class balanced. 
    Refer to README file for a better explanation of simulations. 
    """
    def getSimulatedData(self, track_ranges, num_samples, percent_features, percent_samples):
        min_all_tracks = track_ranges["min"]
        max_all_tracks = track_ranges["max"]
        num_tracks = len(min_all_tracks)

        simulated_samples = np.empty((num_samples, num_tracks))

        #Function to generate random data (after getting the range for each feature)
        for i in range(0, num_tracks):
            simulated_track = np.random.uniform(min_all_tracks[i], max_all_tracks[i], num_samples)
            simulated_samples[:, i] = simulated_track

        #generate random exactly class balanced labels for the data
        zeros = np.zeros(math.floor(num_samples/2))
        ones = np.ones(num_samples - len(zeros))
        labels = np.concatenate((zeros, ones), axis = 0)
        np.random.shuffle(labels)

        #Randomly pick some samples and some features from the simulated random data and add some signal to these values. 
        if(percent_features > 0 and percent_samples > 0):
            num_track_with_signal = math.floor((percent_features * num_tracks)/100)
            num_samples_with_signal = math.floor((percent_samples * num_samples)/100)

            # print(f"For feature percent: {percent_features}, the number of tracks with signal : {num_track_with_signal}")
            # print(f"For sample percent: {percent_samples}, the number of samples with signal : {num_samples_with_signal}")
            track_indices = random.sample(range(0, num_tracks), num_track_with_signal) #Randomly choose some tracks to add signal to
            sample_indices = random.sample(range(0, num_samples), num_samples_with_signal) #Randomly choose some samples to add signal to

            # print(f"Num augmented tracks: {len(track_indices)} and num aug samples : {len(sample_indices)}")

            #For each track, replace some sample with augumented signal - if the sample has a positive label, then make the value slighty higher than the highest value
            #If the sample is negative, the augumented value will be 1 less than the lowest value for the track. 
            pos_indices = np.where(labels == 1)
            pos_indices_to_replace = np.intersect1d(sample_indices, pos_indices)

            neg_indices = np.where(labels == 0)
            neg_indices_to_replace = np.intersect1d(sample_indices, neg_indices)
            
            for track_index in track_indices:
                #Replace positive samples for this track
                #If the percentage augmentation is too low, there all indices picked for augmentation could be negative. In that case, these lines throw an error. 
                if(len(pos_indices_to_replace) > 0): 
                    max_track = max_all_tracks[track_index]
                    replacement_pos_val = random.uniform(max_track, 1.1 * max_track)
                    simulated_samples[pos_indices_to_replace, track_index] = replacement_pos_val

                #Replace negative samples for this track
                #If the percentage augmentation is too low, there all indices picked for augmentation could be positive. In that case, these lines throw an error.
                if(len(neg_indices_to_replace) > 0):
                    min_track = min_all_tracks[track_index]         
                    replacement_neg_val = random.uniform(0.9 * min_track, min_track) #Replace with a value slightly lower than the lowest value of that track
                    simulated_samples[neg_indices_to_replace, track_index] = replacement_neg_val

        return simulated_samples, labels

    # def oldFnAddingControls():
    #     #Based on numSamples and numFeatures, add signal to the data. 
    #     positiveControlValueMin = 0.5
    #     positiveControlValueMax = 1.5
    #     negativeControlValueMin = -1.5
    #     negativeControlValueMax = -0.5

    #     for i, single_sample in enumerate(samples):
    #         if labels[i] == 1:
    #             replacement_val = random.uniform(positiveControlValueMin, positiveControlValueMax)
    #         else:
    #             replacement_val = random.uniform(negativeControlValueMin, negativeControlValueMax)

    #         for j in featureIndicesToReplace:
    #             single_sample[j] = replacement_val
            
    #         samples[i] = single_sample

    #     # #The enoded enformer output only has 20 features. Append 0's for the rest of the features so that the dense layer matrix size is consistent with input 
    #     # nrows, ncols = encoded_enformer_output.shape
    #     # zeroes = torch.zeros(nrows, 10606)
    #     # encoded_enformer_output = torch.cat((encoded_enformer_output, zeroes), axis = 1)
    


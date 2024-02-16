'''
This file has functions to get the Enformer output for all cfDNA coordiante files in a given directory. The steps
involved are as follows
1. Fetch the one hot encoded sequence for each sample in the coordinate files in the coordinate store directory (refer
    to sequenceDataset class for more details on this process)
2. Get Enformer predictions for the one hot encoded sequence.
3. Store the output in H5PY files.
'''

import numpy as np
import torch
import pandas as pd

from enformer_pytorch import Enformer
from torch.utils.data import DataLoader

import h5py
import sys

sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')

import config
import utils
import sequenceDatasetEnformer as sequenceDataset

import importlib   
import os
import time

importlib.reload(sequenceDataset)
importlib.reload(config)
importlib.reload(utils)

arguments = {}

#File paths
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")
arguments["coordStoreDirectory"] = config.enformerScriptConfigs.get("coordStoreDirectory")
arguments["trainingEnformerOuputStoreFile"] = config.enformerScriptConfigs.get("trainingEnformerOutputStoreFile")
arguments["validationEnformerOuputStoreFile"] = config.enformerScriptConfigs.get("validationEnformerOutputStoreFile")
arguments["testEnformerOuputStoreFile"] = config.enformerScriptConfigs.get("testEnformerOutputStoreFile")
arguments["trainingMetadata"] = config.enformerScriptConfigs.get("trainingEnformerMetadata")
arguments["validationMetadata"] = config.enformerScriptConfigs.get("validationEnformerMetadata")
arguments["testMetadata"] = config.enformerScriptConfigs.get("testEnformerMetadata")

#Enformer output model hyperparameters
arguments["enformerBatchSize"] = config.enformerScriptConfigs.get("enformerBatchSize")
arguments["enformerNumberOfWorkers"] = config.enformerScriptConfigs.get("enformerNumberOfWorkers")

#General configs
arguments["file_sharing_strategy"] = config.enformerScriptConfigs.get("fileSharingStrategy")
arguments["enformerOutputFileCompression"] = config.enformerScriptConfigs.get("enformerOutputFileCompression")
arguments["enformerOutputFileChunkSize"] = config.enformerScriptConfigs.get("enformerOutputFileChunkSize")

#Datasets
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")
arguments["trainingEnformerOutputDatasetName"] = config.datasetNames.get("trainingEnformerOutput")
arguments["validationEnformerOutputDatasetName"] = config.datasetNames.get("validationEnformerOutput")
arguments["testEnformerOutputDatasetName"] = config.datasetNames.get("testEnformerOutput")

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Get enformer predictions for a given sequence 
Inputs: 
1. enformer_model - The loaded and initialized pre-trained Enformer model 
2. sequence - One hot encoded sequence (the size of this sequence should match enformer input size). Size is (nb * 196607 * 4)
    where nb is the batch size, 196607 is the fixed sequence length expected by Enformer and 4 is due to one hot encoding. 
3. bins - The bins from the Enformer output that would match the original sequence 
    (backstory - the original cfDNA sequence
    coords are extended on both sides to reach Enformer input length. Enformer predictions for the whole input sequence
    are partitioned into bins, each bin containing one prediction for a 200kb segment of the input sequence. From the method, 
    we only need to return the predictions for the bins that correspond to the original sequence (before extension)). 
    Depending on the location of the original sequence in the extended coords, determine which bins from Enformer output 
    would correspond to the original sequence and give these bins as a tuple of two as input to the function
4. ntracks - Number of tracks from Enformer output that correspond to human (only these tracks are returned from the function)
'''
def getEnformerPredictions(enformer_model, sequence, bins, ntracks):
    # print("Inside get enformer prediction !!")
    with torch.no_grad():
        startBins = bins[0]
        endBins = bins[1]

        #For each output from enformer, get the right bin. 
        full_enformer_output = enformer_model(sequence)['human']
    
    #the enformer prediction is still in the GPU (since we sent the enformer model and one hot encoded sequence to the GPU. Numpy arrays are not supported in the GPU(GPU probably supports only tensors). So we pass the enformer prediction to CPU and convert it into a numpy array.
    #Detach is used to remove the gradients from the predictions. Gradients are similar to the weights of the model. In our case, we are only interested in the predictions and not the model training, so we remove the gradients to save space.
    full_enformer_output = full_enformer_output.detach().cpu()
    final_enformer_output = torch.empty(1, 2, ntracks)
    dims = full_enformer_output.shape

    #Filter out only the required bins from Enformer output
    for i in range(dims[0]):
        #endBin + 1 because of the way torch index based slicing works. x[:, 1:3, :] will give the 1st and 2nd index
        #So the end index in slicing should always be 1 greater than the last index we want. 
        # print(f"For iteration {i}, startBins is {startBins[i]} and endBin is {endBins[i]}")
        single_sample_output = full_enformer_output[i, int(startBins[i]):int(endBins[i]) + 1, :]
        nrows, ncols = single_sample_output.shape
        if(nrows == 0 or ncols == 0):
            print(f"One of the dimensions for single sample output is 0. Here are the details : Shape : {single_sample_output.shape}, Full enformer output shape: {full_enformer_output.shape}, startBin: {startBins[i]} and Endbin: {endBins[i]}")
        single_sample_output = single_sample_output.reshape(1, 2, ntracks)
        final_enformer_output = torch.cat([final_enformer_output,single_sample_output], dim=0)
    
    #The 1st value in the final enformer output is the empty tensor we created for concatenation purposes. 
    final_enformer_output = final_enformer_output[1:, :, :]
    # print(f"Shape of the enformer prediction after taking bins is {final_enformer_output.shape}")
    # print(f"Printing the output shape from enformer {pretrained_output.shape}", flush=True)

    #Combine enformer outputs from 2 bins into a single long output. Each bin, each track is a feature. So the total
    #number of features for training is num_bins * num_tracks_per_bin. All features can be combined in a
    #single 1D tensor array. The other dimension will be the batch size.
    # print("Finished enformer prediction")
    batch_size, nbins, ntracks = final_enformer_output.shape

    #TODO - add logic to also average the bin outputs instead of concatenating, based on user input. 
    final_enformer_output = torch.reshape(final_enformer_output, (batch_size, nbins * ntracks))
    return final_enformer_output
        

'''
This is a dual purpose function. If you pass createDataset as True, then H5PY file will be created and dataset will be 
initialized. If createDataset is false, then the enformerOutputToStore and labelsToStore need to be passed. 

Inputs: 
1. sampleType: can be training/validation or test. Separate H5PY datasets are created for each of the three sets. 
2. numSamples: number of samples to store 
3. numEnformerOuputSingleSample: number of columns in the enformer output to store (here 5313 (number of tracks) * 2) 
4. createDataset - storing this dataset for the first time, then  pass this flag as true. New h5py dataset will be created. 
                Otherwise will be appended to the existing dataset
5. h5_file - Pass false if creating for the first time. Otherwise h5py file object 
6. enformerOutputToStore - Enformer output data to be stored to the h5py file. False if using the function to just 
    create dataset
7. labelsToStore - Labels to store. Again pass false if using function to create dataset. 
8. currentIndex - By default, if storing to an existing dataset the contents are replaced with the new data. 
    To prevent this, story with indices that are increasing with every sample store. Pass the current index from the 
    previous storing iteration, so storing can happen from next index onwards. 

Output - If in createDataset mode, returns the created h5py file. Otherwise, returns the current index. 
'''
def storeAsH5pyFile(sampleType, numSamples, numEnformerOuputSingleSample, createDataset = False, h5_file = False, 
                    enformerOutputToStore=False, labelsToStore=False, currentIndex = False):
   
   enformerOutputDatasetName = arguments[sampleType + "EnformerOutputDatasetName"]
   labelsDatasetName = arguments[sampleType + "LabelsDatasetName"]
   enformerOutputFilePath = arguments[sampleType + "EnformerOuputStoreFile"]
   
   #If we opening the H5PY file for the 1st time then create the dataset and return the file. 
   if createDataset: 
      print("This is the 1st time. Inside createDataset")

      if h5_file == False:
         h5_file = h5py.File(enformerOutputFilePath, "w-")

      h5_file.create_dataset(enformerOutputDatasetName,  (numSamples, numEnformerOuputSingleSample),
                                    compression="gzip", compression_opts=arguments["enformerOutputFileCompression"],
                                      chunks = (arguments["enformerOutputFileChunkSize"], numEnformerOuputSingleSample))
      h5_file.create_dataset(labelsDatasetName, (numSamples, 1), compression="gzip", 
                             compression_opts=arguments["enformerOutputFileCompression"], 
                             chunks = (arguments["enformerOutputFileChunkSize"], 1))
      return(h5_file)

   else:
      sizeOfOutputToStore = len(labelsToStore)
      endIndex = currentIndex + sizeOfOutputToStore
      h5_file[enformerOutputDatasetName][(currentIndex):(endIndex),:] = enformerOutputToStore
      h5_file[labelsDatasetName][(currentIndex):(endIndex),:] = labelsToStore
      return endIndex

'''
For each enformer output stored to h5py file, this function stores the coordinateFile path and the index within the 
coordinate file into a CSV file, to ensure traceability. 
Input - sampleType - training/validation or test
filepathData - filePath for the sample 
indexData - index within the file for the sample. 
'''
def storeMetadataAsCsv(sampleType, filepathData, indexData):
   metadataFileKey = sampleType + "Metadata"
   metadataFilePath = arguments[metadataFileKey]
   metadata = pd.DataFrame({'og_file': filepathData, 'indexInFile':indexData})
   print(f"Shape of metadata df after all batches are done is {metadata.shape}")
   print(f"Storing metadata as CSV, the metadata file path is {metadataFilePath}")
   metadata.to_csv(metadataFilePath, sep='\t', index=False)
   
'''
To prevent index problems due to parallel processing. 
'''
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])

'''
The function that puts all individual aspects of storing Enformer output together. Iterates through sequenceDataset getting 
the one hot encoded sequence, calls functions to get Enformer predictions and store as H5PY files. 

'''
#The function returns 2 numpy arrays. The 1st numpy array is the enformer output for all cfdna fragments. The second numpy array is the array of labels for all cfDNA fragments.
def storeEnformerOutput(sampleType, h5_file = False):
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])

    nbins = 2
    ntracks = 5313

    #Set the model to eval mode first and then send it to cuda. This prevents the GPU node from running out of memory.
    enformerModel = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_checkpointing = True).eval()
    enformerModel = enformerModel.to(device)
    
    enformerInputDataset = sequenceDataset.SequenceDataset(sampleType)
    enformerInputDataloader = DataLoader(enformerInputDataset, batch_size=arguments["enformerBatchSize"], 
                                        num_workers=arguments["enformerNumberOfWorkers"],
                                        shuffle=True, worker_init_fn=set_worker_sharing_strategy)
    
    numSamples = len(enformerInputDataset)

    #Create the datasets for storing enformer output. 
    h5_file = storeAsH5pyFile(sampleType, numSamples, nbins * ntracks, True, h5_file)
    filepath_data, index_data = [], []
    currentH5Index = 0

    for i, data in enumerate(enformerInputDataloader):
        
        #Store the filepath and the index within file to a separate CSV file. This is to ensure that we are able to locate the sample
        #so we can access the metadata(from original coordinate bed file) associated with the sample. 
        #filepath and index should have all the samples data from this batch. 
        encodedSequence, label, bins, filepath, indexWithinFile, _ = data

        filepath_data.extend(filepath)
        index_data.extend(indexWithinFile)

        # print(f"Printing the shape of the encoded sequence {encodedSequence.shape}", flush = True)
        # print(f"Printing the shape of label {label.shape}")
        encodedSequence = encodedSequence.to(device)
        
        #Will be of the shape [batch_size * 10626]
        enformerPrediction = getEnformerPredictions(enformerModel, encodedSequence, bins, ntracks).detach().cpu().numpy()
        
        #The data is getting too big to load, round off enformer predictions to 3 decimal places. 
        enformerPrediction = np.around(enformerPrediction, decimals=3)
        
        """
        H5 file contents are updated every batch. To ensure that the contents are not overwritten every batch, store with indices. 
        The indices given are ascending order numbers starting from 0, this ensures that the shuffled order is maintained while storing in H5PY file. 
        """
        currentH5Index = storeAsH5pyFile(sampleType, numSamples, nbins * ntracks, False, h5_file, enformerPrediction, label, currentH5Index)
        print(f"Finished processing batch {i}. The number of samples stored in H5PY file so far is {currentH5Index}", flush = True)

    h5_file.close()

    #Store the filename and index within the file for each sample as a CSV file for later use. 
    storeMetadataAsCsv(sampleType, filepath_data, index_data)

'''
Verify if the newly created H5PY files are correct. The following assertions are done 
    1. Total share of enformer output file shoud be [num_samples_coordinate_files * 10626]
    2. Number of positives and negatives in enformer file = number of positives in the coordinate store directory
    3. Total shape of the enformer tracks 
'''
def verifyStoredEnformerTracks():

    coordsDir = arguments["coordStoreDirectory"]
    sampleCounts = {}
    sampleCounts["training"] = [0, 0]
    sampleCounts["validation"] = [0,0]
    sampleCounts["test"] = [0,0]

    for filename in os.listdir(coordsDir):
        filepath = os.path.join(coordsDir, filename)
        with h5py.File(filepath, 'r') as f:
            for sampleType in ["training", "validation", "test"]:
                labelsDataset = sampleType + "Labels"
                labels = f[labelsDataset][:]
                sampleCounts[sampleType][0] += (labels == 1).sum()
                sampleCounts[sampleType][1] += (labels == 0).sum()
    
    x = sampleCounts["training"]
    y = sampleCounts["validation"]
    z = sampleCounts["test"]

    print(f"Finished going over the coordinate files, the numbers are {x}, {y} and {z}")

    outputFilesDict = {}
    # outputFilesDict["training"] = arguments["trainingEnformerOuputStoreFile"]

    outputFilesDict["validation"] = arguments["validationEnformerOuputStoreFile"]
    # outputFilesDict["test"] = arguments["testEnformerOuputStoreFile"]
    
    for sampleType, file in outputFilesDict.items(): 
        with h5py.File(file, 'r') as f:
            enformerOutputDataset = sampleType + "EnformerOutput"
            labelsDataset = sampleType + "Labels"
            enformerDataShape = f[enformerOutputDataset][:].shape
            labels = f[labelsDataset][:]

            print(f"Enformer output shape is {enformerDataShape}")
            #Assertion 1 Verify that shape of enformer output is as expected: 
            assert enformerDataShape[0] == sampleCounts[sampleType][0] + sampleCounts[sampleType][1], (f"The total number of samples in enformer output"+
                                                                                                       f" file({enformerDataShape[0]}) does not match the "+
                                                                                                       "total samples in coordinate store directory"
                                                                                                       f"({sampleCounts[sampleType][0] + sampleCounts[sampleType][1]})")
            assert enformerDataShape[1] == 10626, (f"The number of enformer tracks in the output file({enformerDataShape[1]}) "+
                                                    "for samples is not 10626 !!")

            #Assertion - 2 Verify that the number of positives and negatives match 
            numPositives = (labels == 1).sum()
            numNegatives = (labels == 0).sum()
            print(f"Num positives and negatives in enformer for sample {sampleType} are {numPositives} and {numNegatives}")
            print(f"Num pos and neg in coord for sampleType {sampleType} are {sampleCounts[sampleType][0]} and {sampleCounts[sampleType][1]}")

            assert numPositives == sampleCounts[sampleType][0], (f"The number of positives in enformer file({numPositives}) "+
                                                                 f"does not match the original positives {sampleCounts[sampleType][0]}")
            assert numNegatives == sampleCounts[sampleType][1], (f"The number of negatives in enformer file({numNegatives}) "+
                                                                 f"does not match the original negatives {sampleCounts[sampleType][1]}")
            
if __name__ == '__main__':
    print(f"Start time is {time.time()}")
    # storeEnformerOutput("training")
    storeEnformerOutput("validation")
    # verifyStoredEnformerTracks()
    # storeEnformerOutput("test")
    print(f"End time is {time.time()}")

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import h5py
import sys
import torch

sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')

import config
import utils
import sequenceDataset

import importlib   
import os

importlib.reload(sequenceDataset)
importlib.reload(config)
importlib.reload(utils)

arguments = {}
#input paths
arguments["coordStoreDirectory"] = config.enformerScriptConfigs.get("coordStoreDirectory")
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")
arguments["trainingEnformerMetadata"] = config.enformerScriptConfigs.get("trainingEnformerMetadata")
arguments["validationEnformerMetadata"] = config.enformerScriptConfigs.get("validationEnformerMetadata")
arguments["testEnformerMetadata"] = config.enformerScriptConfigs.get("testEnformerMetadata")

#output file paths
arguments["trainingEncodedSequenceFilePath"] = config.filePaths.get("trainingEncodedSequenceFilePath")
arguments["validationEncodedSequenceFilePath"] = config.filePaths.get("validationEncodedSequenceFilePath")
arguments["testEncodedSequenceFilePath"] = config.filePaths.get("testEncodedSequenceFilePath")

#General configs
arguments["cnnInputSequenceSize"] = config.modelHyperParameters.get("modelInputSequenceSize")
arguments["runWithControls"] = config.modelGeneralConfigs.get("runWithControls")
arguments["batchSize"] = config.modelHyperParameters.get("batchSize")
arguments["numWorkers"] = config.modelHyperParameters.get("numberOfWorkers")
arguments["usePaddingForCnn"] = config.modelGeneralConfigs.get("usePaddingForCnn")


#Dataset names
arguments["trainingCoordsDatasetName"] = config.datasetNames.get("trainingCoords")
arguments["validationCoordsDatasetName"] = config.datasetNames.get("validationCoords")
arguments["testCoordsDatasetName"] = config.datasetNames.get("testCoords")
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")
arguments["trainingEncodedSequenceDatasetName"] = config.datasetNames.get("trainingEncodedSequence")
arguments["validationEncodedSequenceDatasetName"] = config.datasetNames.get("validationEncodedSequence")
arguments["testEncodedSequenceDatasetName"] = config.datasetNames.get("testEncodedSequence")
arguments["trainingSequenceLengthDatasetName"] = config.datasetNames.get("trainingSequenceLength")
arguments["validationSequenceLengthDatasetName"] = config.datasetNames.get("validationSequenceLength")
arguments["testSequenceLengthDatasetName"] = config.datasetNames.get("testSequenceLength")


#Get the sequence dataset, but instead of iterating through the whole directory, only pick one sample
#given a file name and the index within the file. 
class SequenceDatasetForEncoding(Dataset):
    def __init__(self, sampleType, metadataDf):
        self.sampleType = sampleType
        self.metadataDf = metadataDf

    def __getitem__(self, index):
        filepath_in_metadata = self.metadataDf["og_file"].iloc[index]
        """
        The metadatadf has the path of the file at the time of creation of the metadata CSV file. Get the filename 
        from the path and generate the new file path from the current coordinate store directory path
        """
        filename_in_metadata = os.path.basename(filepath_in_metadata)
        filepath = os.path.join(arguments["coordStoreDirectory"], filename_in_metadata)
        indexInFileStr = self.metadataDf["indexInFile"].iloc[index] #Convert the index tensor into an int

        #The indexInFile field looks like this tensor(1234). Extract only the value 1234 from the whole string. 
        indexInFile = int(indexInFileStr[indexInFileStr.find("(")+1: indexInFileStr.find(")")])

        with h5py.File(filepath, 'r') as f:
            h5pyCoordsDatasetName = arguments[self.sampleType + "CoordsDatasetName"]
            coord = f[h5pyCoordsDatasetName][indexInFile]
            #Each sample should have only one label, it should be a single value instead of a numpy 1D array.The [0] is to make it a single value instead of a numpy array.
            # label = f['trainingLabels'][index][0]
            h5pyLabelsDatasetName = arguments[self.sampleType + "LabelsDatasetName"]
            label = f[h5pyLabelsDatasetName][:][indexInFile]
            
        sequenceOutputLength = arguments["cnnInputSequenceSize"]
        encoded_input_sequence, _ , og_sequence_length = utils.getOneHotEncodedSequenceFromCoordinates(coord, arguments["refGenomePath"],
                                                                                          sequenceOutputLength, arguments["usePaddingForCnn"])
        assert encoded_input_sequence.shape == (sequenceOutputLength, 4), f"One of the samples did not have the right dimensions({(sequenceOutputLength, 4)}). The sample index is {index}, shape is {encoded_input_sequence.shape}, filepath is {filepath} and index within the file is {indexInFile}"
        
        return encoded_input_sequence, label, og_sequence_length

    def __len__(self):
        return len(self.metadataDf)

def storeAsH5pyFile(sampleType, numSamples, createDataset=False, h5_file=False, sequenceToStore=False, 
                    labelsToStore=False, og_sequence_length = False, currentIndex=False):
    encodedSequenceDatasetName = arguments[sampleType + "EncodedSequenceDatasetName"]
    labelsDatasetName = arguments[sampleType + "LabelsDatasetName"]
    sequenceLengthDatasetName = arguments[sampleType + "SequenceLengthDatasetName"]
    encodedSequenceFilePath = arguments[sampleType + "EncodedSequenceFilePath"]
    sequence_length = arguments["cnnInputSequenceSize"]
    
    #If we opening the H5PY file for the 1st time then create the dataset and return the file. 
    if createDataset: 
        print("This is the 1st time. Inside createDataset")

        if h5_file == False:
            h5_file = h5py.File(encodedSequenceFilePath, "w-")

        h5_file.create_dataset(encodedSequenceDatasetName,  (numSamples, sequence_length, 4),
                                        compression="gzip", compression_opts=8, chunks = (200, sequence_length, 4))
        h5_file.create_dataset(labelsDatasetName, (numSamples, 1), compression="gzip", compression_opts=8, chunks = (200, 1))
        h5_file.create_dataset(sequenceLengthDatasetName, (numSamples, 1), compression="gzip", compression_opts=8, chunks = (200, 1))
        return(h5_file)

    else:
        sizeOfOutputToStore = len(labelsToStore)
        og_sequence_length = torch.reshape(og_sequence_length, (sizeOfOutputToStore, 1))
        endIndex = currentIndex + sizeOfOutputToStore
        h5_file[encodedSequenceDatasetName][(currentIndex):(endIndex),:, :] = sequenceToStore
        h5_file[labelsDatasetName][(currentIndex):(endIndex),:] = labelsToStore
        h5_file[sequenceLengthDatasetName][(currentIndex):(endIndex),:] = og_sequence_length
        return endIndex

def generateOneHotEncodings(sampleType, h5_file = False):    
    #Read Enformer metadata CSV file and get as a dataframe
    metadataFileKey = sampleType + "EnformerMetadata"
    enformerMetadataFile = arguments[metadataFileKey]
    metadataDf = pd.read_csv(enformerMetadataFile, sep = "\t", names=["og_file", "indexInFile"], skiprows=1)
    numSamples = len(metadataDf)

    dataset = SequenceDatasetForEncoding(sampleType, metadataDf)
    dataloader = DataLoader(dataset, batch_size=arguments["batchSize"], num_workers=arguments["numWorkers"], 
                            shuffle=False)

    h5_file = storeAsH5pyFile(sampleType, numSamples, True, h5_file)
    currentH5Index = 0
    
    for i, data in enumerate(dataloader):
        encoded_sequence, label, og_sequence_length = data
        print(f"Shape of the encoded sequence is {encoded_sequence.shape}")

        """
        H5 file contents are updated every batch. To ensure that the contents are not overwritten every batch, store with indices. 
        The indices given are ascending order numbers starting from 0, this ensures that the shuffled order is maintained while storing in H5PY file. 
        """
        currentH5Index = storeAsH5pyFile(sampleType, numSamples, False, h5_file, encoded_sequence, label, og_sequence_length, currentH5Index)
        print(f"Finished processing batch {i}. The number of samples stored in H5PY file so far is {currentH5Index}", flush = True)
    
if __name__ == '__main__':
    # generateOneHotEncodings("training")
    generateOneHotEncodings("validation")
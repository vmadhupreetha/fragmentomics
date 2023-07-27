import pysam
import numpy as np
import torch
import pandas as pd

from enformer_pytorch import Enformer
from torch.utils.data import Dataset, DataLoader

import h5py
import sys

sys.path.insert(0, '/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')

import config
import utils

import importlib
import os

importlib.reload(config)
importlib.reload(utils)

arguments = {}
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")
arguments["coordStoreDirectory"] = config.filePaths.get("coordStoreDirectory")
arguments["trainingEnformerOuputStoreFile"] = config.filePaths.get("trainingEnformerOutputStoreFile")
arguments["validationEnformerOuputStoreFile"] = config.filePaths.get("validationEnformerOutputStoreFile")
arguments["testEnformerOuputStoreFile"] = config.filePaths.get("testEnformerOutputStoreFile")
arguments["trainingMetadata"] = config.filePaths.get("trainingMetadata")
arguments["validationMetadata"] = config.filePaths.get("validationMetadata")
arguments["testMetadata"] = config.filePaths.get("testMetadata")
arguments["file_sharing_strategy"] = config.modelGeneralConfigs.get("fileSharingStrategy")
device = "cuda" if torch.cuda.is_available() else "cpu"


def getOneHotEncodedSequenceFromCoordinates(coord):
    referenceGenome = pysam.FastaFile(arguments["refGenomePath"])
    coords = (coord[0].decode('UTF-8'), int(coord[1]), int(coord[2]))
    # Get surrounding sequence for feeding into enformer.
    (extendedCoordsEnformer, bins) = utils.getCoordsAndBin(coords, referenceGenome)

    # Get the raw sequence using the coordinates and the reference genome.
    cfDnaFragment = utils.getSequenceFromCoord(referenceGenome, extendedCoordsEnformer)

    # One hot encode sequence
    encodedFragment = utils.oneHotEncodeSequence(cfDnaFragment)

    encoded_input_sequence = torch.tensor(np.float32(encodedFragment))
    # encoded_input_sequence = encoded_input_sequence.to\(device)
    return encoded_input_sequence, bins


# It is silly to loop through all files, create a master numpy array and fetch the right index everytime we are calling get_item method
# Store the range of indices for each file in a dictionary. Only open the concerned file when get_item is called for an index.
def createFileNameIndexList(sampleType):
    startIndexList = []
    fileNamesList = []

    coordFilesDirectory = os.fsencode(arguments["coordStoreDirectory"])
    currentIndex = 0
    for file in os.listdir(coordFilesDirectory):
        filename = os.fsencode(file).decode("utf-8")
        startIndexList.append(currentIndex)
        fileNamesList.append(filename)

        numItemsFile = getNumberLinesInFile(filename, sampleType)
        currentIndex = currentIndex + numItemsFile

    print(f"Total number of samples in all files combined is {currentIndex}")
    return startIndexList, fileNamesList


def getNumberLinesInFile(filename, sampleType):
    filePath = os.path.join(arguments["coordStoreDirectory"], filename)
    with h5py.File(filePath, 'r') as f:
        coordsDatasetName = sampleType + "Coords"
        trainingSamples = f[coordsDatasetName][:]
        numSamples = len(trainingSamples)
        return numSamples


"""
Imagine samples from all the files are clubbed together. Given an index belonging to the collection of all samples (from all)
files clubbed together, the index could belong to any file. startIndexList is a list of the start index of all the 492 files. 
Iterate over this startIndexList and find out which position the indexToFind becomes greater than the startIndex. 
The file for which index started at this startIndex will be the one containing the indexToFind. So just get the position of the 
startIndex which is just below the indexToFind and check in the fileNamesList for this position. 
"""


def getFilePositionFromIndex(startIndexList, indexToFind):
    for i, index in enumerate(startIndexList):
        if (indexToFind < index):
            # This means we moved to the next file, so we have to pick the i before that
            filePosition = i - 1
            return filePosition

    # this scenario will never occur unless indexToFind is a negative value.
    return i


def getEnformerPredictions(enformer_model, sequence, bins, ntracks):
    # print("Inside get enformer prediction !!")
    with torch.no_grad():
        startBins = bins[0]
        endBins = bins[1]

        # For each output from enformer, get the right bin.
        full_enformer_output = enformer_model(sequence)['human']

    # the enformer prediction is still in the GPU (since we sent the enformer model and one hot encoded sequence to the GPU. Numpy arrays are not supported in the GPU(GPU probably supports only tensors). So we pass the enformer prediction to CPU and convert it into a numpy array.
    # Detach is used to remove the gradients from the predictions. Gradients are similar to the weights of the model. In our case, we are only interested in the predictions and not the model training, so we remove the gradients to save space.
    full_enformer_output = full_enformer_output.detach().cpu()
    final_enformer_output = torch.empty(1, 2, ntracks)
    dims = full_enformer_output.shape

    for i in range(dims[0]):
        # endBin + 1 because of the way torch index based slicing works. x[:, 1:3, :] will give the 1st and 2nd index
        # So the end index in slicing should always be 1 greater than the last index we want.
        # print(f"For iteration {i}, startBins is {startBins[i]} and endBin is {endBins[i]}")
        single_sample_output = full_enformer_output[i, int(startBins[i]):int(endBins[i]) + 1, :].reshape(1, 2, ntracks)
        # print(f"Shape of single sample enformer output is {single_sample_output.shape}")
        final_enformer_output = torch.cat([final_enformer_output, single_sample_output], dim=0)

    # The 1st value in the final enformer output is the empty tensor we created for concatenation purposes.
    final_enformer_output = final_enformer_output[1:, :, :]
    # print(f"Shape of the enformer prediction after taking bins is {final_enformer_output.shape}")
    # print(f"Printing the output shape from enformer {pretrained_output.shape}", flush=True)

    # Combine enformer outputs from 2 bins into a single long output. Each bin, each track is a feature. So the total
    # number of features for training is num_bins * num_tracks_per_bin. All features can be combined in a
    # single 1D tensor array. The other dimension will be the batch size.
    # print("Finished enformer prediction")
    batch_size, nbins, ntracks = final_enformer_output.shape

    final_enformer_output = torch.reshape(final_enformer_output, (batch_size, nbins * ntracks))
    return final_enformer_output


class EnformerInputDataset(Dataset):
    def __init__(self, sampleType):
        self.sampleType = sampleType
        self.startIndexList, self.fileNamesList = createFileNameIndexList(sampleType)

    def __getitem__(self, index):
        filePosition = getFilePositionFromIndex(self.startIndexList, index)
        filename = self.fileNamesList[filePosition]
        indexWithinFile = index - self.startIndexList[filePosition]

        filepath = os.path.join(arguments["coordStoreDirectory"], filename)

        with h5py.File(filepath, 'r') as f:
            h5pyCoordsDatasetName = self.sampleType + "Coords"
            coord = f[h5pyCoordsDatasetName][indexWithinFile]

            # Each sample should have only one label, it should be a single value instead of a numpy 1D array.The [0] is to make it a single value instead of a numpy array.
            # label = f['trainingLabels'][index][0]
            h5pyLabelsDatasetName = self.sampleType + "Labels"
            label = f[h5pyLabelsDatasetName][:][indexWithinFile]

            encoded_input_sequence, bins = getOneHotEncodedSequenceFromCoordinates(coord)

            # For some cases, the coordinates look fine, but the sequence fetched from the fasta file has size 0.
            # If we pass such samples to enformer for predictions, we get Einops error, due to dimension mismatch.
            # try:
            # assert encoded_input_sequence.shape == (196607,4)

            # except:
            #    print(f"One of the samples did not have the right dimensions. The sample index is {index}, shape is {encoded_input_sequence.shape}")
            #    return torch.empty((196607,4)), label

        return encoded_input_sequence, label, bins, filepath, indexWithinFile

    def __len__(self):
        # Total number of samples is the startIndexList of the last file + number of samples in the last file.
        lastStartIndex = self.startIndexList[-1]
        lastFileName = self.fileNamesList[-1]
        numSamplesLastFile = getNumberLinesInFile(lastFileName, self.sampleType)
        totalNumSamples = lastStartIndex + numSamplesLastFile
        return totalNumSamples


# Look into how much h5py content can be compressed. Greater the compression, longer the time needed to read it again.
def storeAsH5pyFile(sampleType, numSamples, numEnformerOuputSingleSample, createDataset=False, h5_file=False,
                    enformerOutputToStore=False, labelsToStore=False, currentIndex=False):
    enformerOutputDatasetName = sampleType + "EnformerOutput"
    labelsDatasetName = sampleType + "Labels"

    enformerOutputFileKey = sampleType + "EnformerOuputStoreFile"
    enformerOutputFilePath = arguments[enformerOutputFileKey]

    # If we opening the H5PY file for the 1st time then create the dataset and return the file.
    if createDataset:
        print("This is the 1st time. Inside createDataset")

        if h5_file == False:
            h5_file = h5py.File(enformerOutputFilePath, "w-")

        h5_file.create_dataset(enformerOutputDatasetName, (numSamples, numEnformerOuputSingleSample),
                               compression="gzip", compression_opts=8, chunks=(200, numEnformerOuputSingleSample))
        h5_file.create_dataset(labelsDatasetName, (numSamples, 1), compression="gzip", compression_opts=8,
                               chunks=(200, 1))
        return (h5_file)

    else:
        sizeOfOutputToStore = len(labelsToStore)
        endIndex = currentIndex + sizeOfOutputToStore
        h5_file[enformerOutputDatasetName][(currentIndex):(endIndex), :] = enformerOutputToStore
        h5_file[labelsDatasetName][(currentIndex):(endIndex), :] = labelsToStore
        return endIndex


def storeMetadataAsCsv(sampleType, filepathData, indexData):
    metadataFileKey = sampleType + "Metadata"
    metadataFilePath = arguments[metadataFileKey]
    metadata = pd.DataFrame({'og_file': filepathData, 'indexInFile': indexData})
    print(f"Shape of metadata df after all batches are done is {metadata.shape}")
    metadata.to_csv(metadataFilePath, sep='\t', index=False)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])


# The function returns 2 numpy arrays. The 1st numpy array is the enformer output for all cfdna fragments. The second numpy array is the array of labels for all cfDNA fragments.
def storeEnformerOutput(sampleType, h5_file=False):
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])

    nbins = 2
    ntracks = 5313

    # Set the model to eval mode first and then send it to cuda. This prevents the GPU node from running out of memory.
    enformerModel = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_checkpointing=True).eval()
    enformerModel = enformerModel.to(device)

    enformerInputDataset = EnformerInputDataset(sampleType)
    enformerInputDataloader = DataLoader(enformerInputDataset, batch_size=8, num_workers=12,
                                         shuffle=True, worker_init_fn=set_worker_sharing_strategy)

    numSamples = len(enformerInputDataset)

    # Create the datasets for storing enformer output.
    h5_file = storeAsH5pyFile(sampleType, numSamples, nbins * ntracks, True, h5_file)
    filepath_data, index_data = [], []
    currentH5Index = 0

    for i, data in enumerate(enformerInputDataloader):
        # print(f"Processing data for batch {i}", flush = True)

        # Store the filepath and the index within file to a separate CSV file. This is to ensure that we are able to locate the sample
        # so we can access the metadata(from original coordinate bed file) associated with the sample.
        # filepath and index should have all the samples data from this batch.
        encodedSequence, label, bins, filepath, indexWithinFile = data

        filepath_data.extend(filepath)
        index_data.extend(indexWithinFile)

        # print(f"Printing the shape of the encoded sequence {encodedSequence.shape}", flush = True)
        # print(f"Printing the shape of label {label.shape}")
        encodedSequence = encodedSequence.to(device)

        # Will be of the shape [batch_size * 10626]
        enformerPrediction = getEnformerPredictions(enformerModel, encodedSequence, bins,
                                                    ntracks).detach().cpu().numpy()
        print(f"Finished processsing batch {i}, enformer prediction shape is {enformerPrediction.shape}", flush=True)

        # The data is getting too big to load, round off enformer predictions to 3 decimal places.
        enformerPrediction = np.around(enformerPrediction, decimals=3)

        """
        H5 file contents are updated every batch. To ensure that the contents are not overwritten every batch, store with indices. 
        The indices given are ascending order numbers starting from 0, this ensures that the shuffled order is maintained while storing in H5PY file. 
        """
        currentH5Index = storeAsH5pyFile(sampleType, numSamples, nbins * ntracks, False, h5_file, enformerPrediction,
                                         label, currentH5Index)
        print(f"The number of samples stored in H5PY file so far is {currentH5Index}", flush=True)

    h5_file.close()

    # Store the filename and index within the file for each sample as a CSV file for later use.
    storeMetadataAsCsv(sampleType, filepath_data, index_data)


def verifyStoredEnformerTracks():
    """
    Assertions to be done for stored enformer tracks
    1. Total share of enformer output file shoud be [num_samples_coordinate_files * 10626]
    1. Number of positives and negatives in enformer file = number of positives in the coordinate store directory
    2. Total shape of the enformer tracks
    """
    coordsDir = arguments["coordStoreDirectory"]
    sampleCounts = {}
    sampleCounts["training"] = [0, 0]
    sampleCounts["validation"] = [0, 0]
    sampleCounts["test"] = [0, 0]

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
            # Assertion 1 Verify that shape of enformer output is as expected:
            assert enformerDataShape[0] == sampleCounts[sampleType][0] + sampleCounts[sampleType][1], (
                        f"The total number of samples in enformer output" +
                        f" file({enformerDataShape[0]}) does not match the " +
                        "total samples in coordinate store directory"
                        f"({sampleCounts[sampleType][0] + sampleCounts[sampleType][1]})")
            assert enformerDataShape[1] == 10626, (
                        f"The number of enformer tracks in the output file({enformerDataShape[1]}) " +
                        "for samples is not 10626 !!")

            # Assertion - 2 Verify that the number of positives and negatives match
            numPositives = (labels == 1).sum()
            numNegatives = (labels == 0).sum()
            print(
                f"Num positives and negatives in enformer for sample {sampleType} are {numPositives} and {numNegatives}")
            print(
                f"Num pos and neg in coord for sampleType {sampleType} are {sampleCounts[sampleType][0]} and {sampleCounts[sampleType][1]}")

            assert numPositives == sampleCounts[sampleType][0], (
                        f"The number of positives in enformer file({numPositives}) " +
                        f"does not match the original positives {sampleCounts[sampleType][0]}")
            assert numNegatives == sampleCounts[sampleType][1], (
                        f"The number of negatives in enformer file({numNegatives}) " +
                        f"does not match the original negatives {sampleCounts[sampleType][1]}")


if __name__ == '__main__':
    storeEnformerOutput("training")
    # storeEnformerOutput("validation")
    # verifyStoredEnformerTracks()
    # storeEnformerOutput("test")

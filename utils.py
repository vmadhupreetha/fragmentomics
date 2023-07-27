import math
import numpy as np
import os
import h5py

import pysam
import torch


# TODO get strand information - sense or antisense
def getSequenceFromCoord(refGenome, coordinate):
    # print(f"Getting sequence for coordinate {coordinate}")
    chromosomeNumber = str(coordinate[0])
    start = int(coordinate[1])
    end = int(coordinate[2])
    modified_chromosome = "chr" + chromosomeNumber
    sequence = refGenome.fetch(modified_chromosome, start, end)
    sequence = sequence.upper()

    # Reverse coordinate DNA. Where do we have +/- information in the bed file ?

    # TODO If the cfDNA fragment belongs to the - strand, then the reference genome's compliment has to be taken
    # and also reverse the sequence string.
    return sequence


def oneHotEncodeSequence(sequence):
    base_encoding_map = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]), "T": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]), "C": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0])
    }
    basesList = list(sequence)
    for index, base in enumerate(basesList):
        encoded_base = base_encoding_map[base]
        basesList[index] = encoded_base

    return np.array(basesList)


def getLengthReferenceGenome(refGenome):
    total_length = 0
    for i in range(1, 22):
        total_length = total_length + refGenome.get_reference_length("chr" + str(i))

    return total_length


"""
cfDNA fragments are only ~200 bps long. So extend the coordinates on either sides. 
But sometimes, we might extend beyond the length of the chromosome. If that happens, 
then extend whatever could not be extended on the other side. If extension happens on the other side, 
then if we take the center bin of the enformer output it may no longer coincide with the cfDNA fragment.
So also calculate the bin shift and return the 2 bins which contain the cfDNA fragment output. 
"""


def getCoordsAndBin(coords, refGenome):
    nbins = 1536  # enformer totally outputs 1536 bins. So 1536/2 bins on either side of the mid point of the cfDNA fragment.
    binSize = 128  # Size of each bin output by enformer.
    enformerInputLength = nbins * binSize  # number of base pairs enformer accepts as input. This is nbinsTotal * sizeBin
    cfdnaFragmentBins = (448, 449)

    chrom = coords[0]
    start = int(coords[1])
    end = int(coords[2])
    mid = math.floor((end + start) / 2)
    lengthChromosome = refGenome.get_reference_length("chr" + str(chrom))

    newStart = mid - enformerInputLength / 2 + 1
    newEnd = mid + enformerInputLength / 2

    if newEnd > lengthChromosome:
        print("End is greater than the chromosome length")
        # Number of full bins that are possible to the right of the mid point of cfDNA fragment.
        nPossibleBins = math.floor((lengthChromosome - mid) / binSize)

        # The number of bins to the left the center of the input sequence should be shifted, to compensate for the
        # lack of length on the right.
        shift = nbins / 2 - nPossibleBins
        newStart = newStart - (shift * binSize)
        newEnd = mid + (nPossibleBins * binSize)
        cfdnaFragmentBins = (cfdnaFragmentBins[0] - shift, cfdnaFragmentBins[1] - shift)

    if newStart < 0:
        print("start is less than 0")
        nPossibleBins = math.floor(mid / binSize)
        print(f"Possible bins {nPossibleBins}")
        # The number of bins to the right the center of the input sequence should be shifted, to compensate for the
        # lack of length on the left.
        shift = nbins / 2 - nPossibleBins
        newEnd = newEnd + (shift * binSize)
        newStart = mid - (nPossibleBins * binSize) + 1
        cfdnaFragmentBins = (cfdnaFragmentBins[0] + shift, cfdnaFragmentBins[1] + shift)

    newCoords = (coords[0], newStart, newEnd)

    return (newCoords, cfdnaFragmentBins)


# Enformer takes around 200kb of input. Each cfDNA fragment is only around 200 bases.
# Take surrounding regions of the reference genome too.
def getCoordsForEnformerOld(coords, refGenome):
    # If we take more than this value, we run out of memory (Process finished with exit code 137 (interrupted by signal 9: SIGKILL))
    enformerInputLength = 196608

    chrom = coords[0]
    start = int(coords[1])
    end = int(coords[2])
    mid = math.floor((end + start) / 2)

    newStart = mid - enformerInputLength / 2 + 1
    newEnd = mid + enformerInputLength / 2
    lengthChromosome = refGenome.get_reference_length("chr" + str(chrom))

    if newStart < 0:
        newStart = 0

    # If padding on the left exceeds the length of reference genome left, then add the extra padding on the right.
    # Similarly if padding on the right exceeds the length of the reference genome, add the extra padding on the left.
    if (newStart < 0):
        extraLength = -(newStart)
        newEnd = newEnd + extraLength
        newStart = 0

    if (newEnd > lengthChromosome):
        extraLength = newEnd - lengthChromosome + 1
        newStart = newStart - extraLength
        newEnd = lengthChromosome - 1

    lengthOfFragment = newEnd - newStart + 1

    try:
        assert lengthOfFragment == enformerInputLength - 1

    except:
        print(
            f"One of the extended coordinate samples generated for enformer does not have the required input length of {enformerInputLength}. The length of coordiantes in question is {lengthOfFragment}")

    newCoords = (coords[0], newStart, newEnd)
    return newCoords


"""
Given a coordinate tuple and a reference genome path, get the one hot encoded sequence from the reference
genome that falls within these coordinates. 
"""


def getOneHotEncodedSequenceFromCoordinates(coord, referenceGenomePath):
    referenceGenome = pysam.FastaFile(referenceGenomePath)
    coords = (coord[0].decode('UTF-8'), int(coord[1]), int(coord[2]))
    # Get surrounding sequence for feeding into enformer.
    (extendedCoordsEnformer, bins) = getCoordsAndBin(coords, referenceGenome)

    # Get the raw sequence using the coordinates and the reference genome.
    cfDnaFragment = getSequenceFromCoord(referenceGenome, extendedCoordsEnformer)

    # One hot encode sequence
    encodedFragment = oneHotEncodeSequence(cfDnaFragment)

    encoded_input_sequence = torch.tensor(np.float32(encodedFragment))
    # encoded_input_sequence = encoded_input_sequence.to\(device)
    return encoded_input_sequence, bins


"""
Loop through all the H5PY files in a given directory. Create a list of starting indices for all the files in the directory
While training neural network, we use an aggregated index system, where indices of all files are combined. 
Given this aggregated index, this list is useful to find out which file the index belongs to, so the sample can be fetched
"""


def createFileNameIndexList(directory, datasetName):
    startIndexList = []
    fileNamesList = []

    coordFilesDirectory = os.fsencode(directory)
    currentIndex = 0
    for filename in os.listdir(coordFilesDirectory):
        startIndexList.append(currentIndex)
        fileNamesList.append(filename)
        filePath = os.path.join(directory, filename)

        numItemsFile = getNumberLinesInFile(filePath, datasetName)
        currentIndex = currentIndex + numItemsFile

    print(f"Total number of samples in all files combined is {currentIndex}")
    return startIndexList, fileNamesList


"""
Given a H5PY file path and the dataset name inside the file, get the number of samples in the file.
"""


def getNumberLinesInFile(h5pyFilePath, datasetName):
    with h5py.File(h5pyFilePath, 'r') as f:
        numSamples = len(f[datasetName][:])
        return numSamples


"""
Given a list of starting indices of all files and an index, find which file this index belongs to 
If indices are aggregated for all files, then startIndexList is an integer list of the indices that mark the beginning of each file 
The output is the position index of the file - out of all the files, which file number has the given index 
"""


def getFilePositionFromIndex(startIndexList, indexToFind):
    for i, index in enumerate(startIndexList):
        if (indexToFind < index):
            # This means we moved to the next file, so we have to pick the i before that
            filePosition = i - 1
            return filePosition

    # this scenario will never occur unless indexToFind is a negative value.
    return -1
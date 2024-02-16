'''
Contains a set of miscellaneous methods for manipulating coordinate files and sequence.
'''

import math
import numpy as np
import os
import h5py

import pysam
import torch

'''
Inputs
1. refGenome - reference genome pysam object 
2. coordinate - tuple of chromosome number, start and end coordinates for a single cfDNA sample. 

Output - DNA sequence corresponding to the given coordinates. 
'''
#TODO If the cfDNA fragment belongs to the - strand, then the reference genome's compliment has to be taken and
# also reverse the sequence string.
def getSequenceFromCoord(refGenome, coordinate):
    # print(f"Getting sequence for coordinate {coordinate}")
    chromosomeNumber = str(coordinate[0])
    start = int(coordinate[1])
    end = int(coordinate[2])
    modified_chromosome = "chr"+ chromosomeNumber
    sequence = refGenome.fetch(modified_chromosome, start, end)
    sequence = sequence.upper()
    return sequence

'''
Input - DNA sequence (ATGCs) 
Output - one hot encoded sequence (0s and 1s) 
'''
def oneHotEncodeSequence(sequence):
    base_encoding_map = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),"T": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]), "C": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0])
    }
    basesList = list(sequence)
    for index, base in enumerate(basesList):
        encoded_base = base_encoding_map[base]
        basesList[index] = encoded_base

    return np.array(basesList)

'''
Input - pysam reference genome object 
Output - length of the ref genome. 
'''
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
def getCoordsForEnformerAndBin(coords, refGenome):
    # nbins = 896 #enformer totally outputs 896 bins. So 896/2 bins on either side of the mid point of the cfDNA fragment. 
    # binSize = 128 #Size of each bin output by enformer. 
    enformerInputLength = 196608
    cfdnaFragmentBins = (448,449)

    # chrom = coords[0]
    start = int(coords[1])
    end = int(coords[2])
    mid = math.floor((end + start)/2)
    # lengthChromosome = refGenome.get_reference_length("chr" + str(chrom))

    newStart = mid - enformerInputLength/2 + 1 
    newEnd = mid + enformerInputLength/2

    '''
    The scenarios where the fragment is too close to the end (and hence the extension has to be done asymetrically on both sides)
    were getting really complicated. Later, these fragments were dropped from the dataset so this asymmetric extension code
    is no longer in use
    '''
    # if newEnd > lengthChromosome:
    #     #Number of full bins that are possible to the right of the mid point of cfDNA fragment. 
    #     nPossibleBins = math.floor((lengthChromosome - mid)/binSize)

    #     #The number of bins to the left the center of the input sequence should be shifted, to compensate for the 
    #     #lack of length on the right. 
    #     shift = nbins/2 - nPossibleBins
    #     newStart = newStart - (shift * binSize)
    #     newEnd = mid + (nPossibleBins * binSize)
    #     cfdnaFragmentBins = (cfdnaFragmentBins[0]- shift, cfdnaFragmentBins[1] - shift)
    
    # if newStart < 0:
    #     nPossibleBins = math.floor(mid/binSize)
    #     #The number of bins to the right the center of the input sequence should be shifted, to compensate for the 
    #     #lack of length on the left.
    #     shift = nbins/2 - nPossibleBins
    #     newEnd = newEnd + (shift * binSize)
    #     newStart = mid - (nPossibleBins * binSize) + 1
    #     cfdnaFragmentBins = (cfdnaFragmentBins[0] + shift, cfdnaFragmentBins[1] + shift)

    newCoords = (coords[0], newStart, newEnd)

    return (newCoords, cfdnaFragmentBins)

'''
The one hot encoded sequence is passed into CNN filters of fixed input size, whereas the original cfDNA fragments are
of varying lengths. Extend the coordinates (similar to extension for Enformer) so that CNN input size is reached. 

Inputs: 
1. coords - tuple of chrom number, start and end coords for a single cfDNA sample. 
2. refGenome - reference genome
3. required_length - CNN filter input length 

Output- 
Tuple (chrom, start and end) of extended coordinates. 
'''
def getExtendedCoordinatesForCnn(coords, refGenome, required_length):
    chrom = coords[0]
    start = int(coords[1])
    end = int(coords[2])
    lengthChromosome = refGenome.get_reference_length("chr" + str(chrom))
    
    mid = math.floor((end + start)/2)
    newStart = mid - math.ceil(required_length/2)
    newEnd = mid + math.floor(required_length/2)

    if newStart < 0:
        print(f"Start became less than 0", flush =True)
        extraLength = -(newStart)
        newEnd = newEnd + extraLength
        newStart = 0
    
    if newEnd > lengthChromosome:
        print(f"End became greater than chrom length", flush = True)
        extraLength = newEnd - lengthChromosome + 1
        newStart = newStart - extraLength
        newEnd = lengthChromosome - 1
    
    lengthOfFragment = newEnd - newStart
    assert lengthOfFragment == required_length
    newCoords = (coords[0], newStart, newEnd)
    return newCoords

"""
Given a coordinate tuple and a reference genome path, get the one hot encoded sequence from the reference
genome that falls within these coordinates. 
"""
def getOneHotEncodedSequenceFromCoordinates(coord, referenceGenomePath, finalSequenceLength, usePaddingForCnn):
    referenceGenome = pysam.FastaFile(referenceGenomePath)
    coords = (coord[0].decode('UTF-8'), int(coord[1]), int(coord[2]))
    og_sequence_length = coords[2] - coords[1]
    bins = []

    """
    cfDNA fragments can be of varying sizes, but if we want to give the fragment as input to a CNN, all inputs must be of the same size 
    There are 2 options - padding with 0's and extending on the reference genome. If finalSequenceLength is "default" padding is done
    Otherwise, the sequence is extended on either side of the reference genome so that final length reaches the finalSequenceLength
    If the coordinates are being fetched for intput into Enformer, then the size of the required intput to Enformer is fixed
    and a separate method is used to fetch these extended coordinates for enformer. 
    """
    if(finalSequenceLength == "enformer"):
        (coords, bins) = getCoordsForEnformerAndBin(coords, referenceGenome)

    elif(usePaddingForCnn != True):
        coords = getExtendedCoordinatesForCnn(coords, referenceGenome, finalSequenceLength)
    
    #Get the raw sequence using the coordinates and the reference genome.
    cfDnaFragment = getSequenceFromCoord(referenceGenome, coords)

    #One hot encode sequence
    encodedFragment = oneHotEncodeSequence(cfDnaFragment)
    if(finalSequenceLength != "enformer" and usePaddingForCnn == True):
        encodedFragment = addPadding(encodedFragment, finalSequenceLength)

    encoded_input_sequence = torch.tensor(np.float32(encodedFragment))
    if(encoded_input_sequence.nelement() == 0 or len(encoded_input_sequence) == 0 or encoded_input_sequence.numel() == 0):
        print(f"Inside getOneHotEncodedSequenceFromCoordinates, the size of encoded input sequence is 0. The coordinates are {coord}")
    return encoded_input_sequence, bins, og_sequence_length

'''
Given a one hot encoded sequence, pad with one hot encoded 0s until the desired length is reached. 
Input - 
1. Sequence - one hot encoded sequence to be passed
2. outputSize - expected output size after padding 
3. padding_value - the value to be padded with (0 by default)
'''
def addPadding(sequence, outputSize, padding_value = 0):
    sequenceLength = sequence.shape[0]
    extraLength = (outputSize - sequenceLength)
    leftPad = math.floor(extraLength/2)
    rightPad = leftPad if extraLength % 2 == 0 else leftPad + 1
    newSequence = np.pad(sequence,((leftPad, rightPad), (0,0)), constant_values = padding_value)
    return newSequence

"""
Loop through all the H5PY files in a given directory. Create a list of starting indices for all the files in the directory
While training neural network, we use an aggregated index system, where indices of all files are combined. 
Given this aggregated index, this list is useful to find out which file the index belongs to, so the sample can be fetched
"""
def createFileNameIndexList(directory, datasetName):
    startIndexList = []
    fileNamesList = []
    
    currentIndex = 0
    for filename in os.listdir(directory):
        startIndexList.append(currentIndex)
        fileNamesList.append(filename)
        numItemsFile = getNumberLinesInFile(directory, filename, datasetName)
        currentIndex = currentIndex + numItemsFile

    print(f"Total number of samples in all files combined is {currentIndex}")
    return startIndexList, fileNamesList

"""
Given a H5PY file path and the dataset name inside the file, get the number of samples in the file.
"""
def getNumberLinesInFile(directory, filename, datasetName):
    h5pyFilePath = os.path.join(directory, filename)
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
      if(indexToFind < index):
         #This means we moved to the next file, so we have to pick the i before that
         filePosition = i-1
         return filePosition
      
   #this scenario will never occur unless indexToFind is a negative value. 
   return -1
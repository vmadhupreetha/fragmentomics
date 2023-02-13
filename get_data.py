import math

import pysam
import numpy as np

"""
Take the bed file path as input. Return a coords tuple, where coords[0] = chromosome number (1st column of bed file),
coords[1] = start index, coords[2] = end index. 
"""
def getCoordinatesFromBedFile(bedfilePath):
    coords = list()

    # First few lines has metadata about the bed file. Skip those lines when populating the coordinates.
    headers_list = ['#chrom', 'start', 'end', 'read_id', 'mapq', 'cigar1', 'cigar2']
    data_line = False

    # TODO remove this code to only consider the 1st 50 lines of bedfile later
    max_count = 50
    count = 0

    with open(bedfilePath) as file:
        for line in file:
            line = line.strip().split()
            if data_line:
                if count == max_count:    # TODO remove this line
                    print(f"Count reached 50, not reading bed file anymore. ")
                    break
                coordTuple = (line[0], line[1], line[2])
                coords.append(coordTuple)
                count = count + 1     # TODO remove this line
            if line == headers_list:
                data_line = True

    #print(f"Coordinates returned from bed file {coords}")
    return coords


def getSequenceFromCoord(refGenome, coordinate):
    #print(f"Getting sequence for coordinate {coordinate}")
    chromosomeNumber = coordinate[0]
    start = int(coordinate[1])
    end = int(coordinate[2])
    #print(f"About to call fetch with the following arguments : {chromosomeNumber} {start} {end}")
    modified_chromosome = "chr"+chromosomeNumber
    sequence = refGenome.fetch(modified_chromosome, start, end)
    sequence = sequence.upper()
    #Reverse coordinate DNA. Where do we have +/- information in the bed file ?
    return sequence


def replaceSnpsIncfdnaFragments(sequence, start, snps):
    #The coordinates for the SNPs are given wrt to the reference genome. The positions in the fragments will be different

    positions = snps[0].split(",")
    base = snps[1].split(",")

    for position, base in zip(positions, base):
        if base in ["A", "T", "G", "C"]:
            seqList = list(sequence)
            positionInFragment = position - start + 1
            seqList[int(positionInFragment)] = base
            sequence = "".join(seqList)

    return sequence

#The output returned by this method is an array of arrays. The inner most array is 0's and 1's encoding.
#The outermost array is the sequence in np.array format.
def oneHotEncodeSequence(sequence):
    base_encoding_map = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),"T": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]), "C": np.array([0.0, 0.0, 0.0, 1.0])
    }
    basesList = list(sequence)
    for index, base in enumerate(basesList):
        encoded_base = base_encoding_map[base]
        basesList[index] = encoded_base

    return np.array(basesList)

    # TODO: Include code for padding


#Enformer takes around 200kb of input. Each cfDNA fragment is only around 200 bases.
#Take surrounding regions of the reference genome too.
def getCoordsForEnformer(coords, refGenome):
    # If we take more than this value, we run out of memory (Process finished with exit code 137 (interrupted by signal 9: SIGKILL))
    enformerInputLength = 196608
    start = int(coords[1])
    end = int(coords[2])

    #TODO get a better method for getting the length of reference genome.
    lengthReferenceGenome = getLengthReferenceGenome(refGenome)

    lengthOfFragment = end - start
    #print(f"Printing length of fragment {lengthOfFragment}")

    extraLength = math.floor((enformerInputLength - lengthOfFragment)/2)

    if(extraLength >= 0):
        newStart = 0
        newEnd = lengthReferenceGenome

        if(start >= extraLength):
            newStart = start - extraLength

        if(end + extraLength <= lengthReferenceGenome):
            newEnd = end + extraLength

    newCoords = (coords[0], newStart, newEnd)
    newLength = int(newCoords[2]) - int(newCoords[1])
    #print(f"New coords length is {newLength}")
    return newCoords

def getLengthReferenceGenome(refGenome):
    total_length = 0
    for i in range(1, 22):
        total_length = total_length + refGenome.get_reference_length("chr" + str(i))

    return total_length

#Tensors can be made only from numerical data. Convert sequence strings to numerical data
def convertToNumbers(sequence):
    sequence_list = list(sequence)

    num_map = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 0}

    for index, base in enumerate(sequence_list):
        sequence_list[index] = num_map[base]

    return np.array(sequence_list)

def getLabelledDataFromTraining():
    fastaFilePath = "/Users/madhupv/Documents/uu_documents/major_research_project/data/hg38.fa"
    refGenome = pysam.FastaFile(fastaFilePath)
    random_snps = tuple(["1, 35, 40", "A, C, G"])

    recipientFilePath = "/Users/madhupv/Documents/uu_documents/major_research_project/data/L69-M6.recipient.frag.bed"
    donorFilePath = "/Users/madhupv/Documents/uu_documents/major_research_project/data/L69-M6.donor.frag.bed"



#Return labelled data for the cfDNA sequences and whether they are deemed as having cancer or not.
def getLabelledCfDnaSequence(coordinatesFilePath, label, refGenome, snps):

    coordsList = getCoordinatesFromBedFile(coordinatesFilePath)

    sequence_data_list = list()
    for coord in coordsList:

        modifiedCoordinates = getCoordsForEnformer(coords=coord, refGenome=refGenome)
        sequence_from_coords = getSequenceFromCoord(refGenome= refGenome, coordinate=modifiedCoordinates)

        startPositionOfFragment = coord[1]
        sequence_with_snps = replaceSnpsIncfdnaFragments(sequence=sequence_from_coords, start=startPositionOfFragment, snps=snps)

        #If we do this one hot encoding, enformer gives an error saying it is expecting double instead of float.
        encoded_sequence = oneHotEncodeSequence(sequence_with_snps)

        # sequence_np = convertToNumbers(encoded_sequence)
        sequence_data_list.append(encoded_sequence)

    print(f"Number of fragments {len(sequence_data_list)}")
    return sequence_data_list

"""
Each numpy array looks like this 
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]]
Only 4 bases are shown here. The numpy array for a one coord will have as many rows as num of sequences. 
The final output is a list of such arrays
"""

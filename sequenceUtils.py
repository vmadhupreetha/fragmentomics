"""
The file has some methods to get the sequence to be used for enformer predictions from the cfdna fragment coordinate
bed files.
"""
import math
import numpy as np

#TODO get strand information - sense or antisense
"""
Returns the DNA sequence corresponding to a set of cfdna fragment coordinates. 

Args: 
1. refGenome(Fastafile) - the reference genome fasta file. 
2. coordinate(tuple) - a tuple of chromosomeNumber(string), start(string) and end coordinates(string) of cfdna fragments

Output(string) - A string of ATGCs - the sequence of the cfdna fragment
"""
def getSequenceFromCoord(refGenome, coordinate):
    #print(f"Getting sequence for coordinate {coordinate}")
    chromosomeNumber = str(coordinate[0])
    start = int(coordinate[1])
    end = int(coordinate[2])
    #print(f"About to call fetch with the following arguments : {chromosomeNumber} {start} {end}")
    modified_chromosome = "chr"+ chromosomeNumber
    sequence = refGenome.fetch(modified_chromosome, start, end)
    sequence = sequence.upper()
    #Reverse coordinate DNA. Where do we have +/- information in the bed file ?

    #TODO If the cfDNA fragment belongs to the - strand, then the reference genome's compliment has to be taken
    #and also reverse the sequence string.
    return sequence

"""
One hot encodes a sequence. 
Args: 
sequence(string) - the DNA sequence of ATGC's to one hot encode. 

Output(2d numpy array): A 2D numpy array where each row is the one hot encoded value for one base pair. The number of rows will be the 
number of bases in the input sequence.  
A sample output for the input ATGCN would be 
1000
0100
0010
0001
0000
"""
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

"""
Given a fasta file of a genome, it iterates through all the chromosomes and sums up the lengths to get the total length 
of the genome in question. 

Args: 
refGenome(fasta file) -> Fasta file of the genome 

Output(integer) -> Length of the input genome
"""
def getLengthReferenceGenome(refGenome):
    total_length = 0
    for i in range(1, 22):
        total_length = total_length + refGenome.get_reference_length("chr" + str(i))

    return total_length

"""
Given a set of input coordinates, the function returns a new set of coordinates of length 196608, with the same center 
as the input coordiantes. Enformer expects the input sequence to be of the length 196608, but cfDNA fragments are 
typically 200 kbps long. So this function extends the coordinates, so that the enformer input length is reached

Args: 
coords(tuple) -> Tuple of 3 values - the chromosome number (string), start(string) and end(string) 
refGenome -> The fasta file of the reference genome 

Output 
tuple of new coordinates containing chromosome number, new start and new end. 
"""
#Enformer takes around 200kb of input. Each cfDNA fragment is only around 200 bases.
#Take surrounding regions of the reference genome too.
def getCoordsForEnformer(coords, refGenome):
    # If we take more than this value, we run out of memory (Process finished with exit code 137 (interrupted by signal 9: SIGKILL))
    enformerInputLength = 196608

    start = int(coords[1])
    end = int(coords[2])
    mid = math.floor((end + start)/2)

    newStart = mid - enformerInputLength/2 + 1
    newEnd = mid + enformerInputLength/2

    #TODO get a better method for getting the length of reference genome.
    lengthReferenceGenome = getLengthReferenceGenome(refGenome)

    #If padding on the left exceeds the length of reference genome left, then add the extra padding on the right.
    #Similarly if padding on the right exceeds the length of the reference genome, add the extra padding on the left.
    if(newStart < 0):
        extraLength = -(newStart)
        newEnd = newEnd + extraLength
        newStart = 0

    if(newEnd > lengthReferenceGenome):
        extraLength = lengthReferenceGenome - end + 1
        newStart = newStart - extraLength
        newEnd = lengthReferenceGenome - 1

    lengthOfFragment = newEnd - newStart + 1
    newCoords = (coords[0], newStart, newEnd)
    return newCoords
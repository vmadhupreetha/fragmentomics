import argparse
from pathlib import Path

import pysam
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import h5py
import math

import torch.optim as optim

from enformer_pytorch import Enformer
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
arguments = {}

def parseInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refGenomePath', type=str, required=True)
    parser.add_argument('--dataFile', type=str, required=True)

    args = parser.parse_args()
    print(f"Type of args is {type(args)}")

    for path in vars(args):
        file = Path(args.__getattribute__(path))
        if not file.exists():
            print(f"The path {path} does not exist")
            raise SystemExit(1)

    arguments["refGenomePath"] = args.refGenomePath
    arguments["dataFile"] = args.dataFile
    print(f"Finished setting the args. The args for the file is now {arguments}")


#TODO get strand information - sense or antisense
def getSequenceFromCoord(refGenome, coordinate):
    #print(f"Getting sequence for coordinate {coordinate}")
    chromosomeNumber = coordinate[0]
    start = int(coordinate[1])
    end = int(coordinate[2])
    #print(f"About to call fetch with the following arguments : {chromosomeNumber} {start} {end}")
    modified_chromosome = "chr"+str(np.int_(chromosomeNumber))
    sequence = refGenome.fetch(modified_chromosome, start, end)
    sequence = sequence.upper()
    #Reverse coordinate DNA. Where do we have +/- information in the bed file ?

    #TODO If the cfDNA fragment belongs to the - strand, then the reference genome's compliment has to be taken
    #and also reverse the sequence sttring.
    return sequence

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

def getLengthReferenceGenome(refGenome):
    total_length = 0
    for i in range(1, 22):
        total_length = total_length + refGenome.get_reference_length("chr" + str(i))

    return total_length

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
    return newCoords

def getEnformerModel():
    enformer_pretrained = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_checkpointing = True)
    enformer_pretrained.to(device)
    enformer_pretrained.eval()
    return enformer_pretrained

def getEnformerPredictions(enformer_model, sequence):
    pretrained_output = enformer_model(sequence)

    # Take only the middle bins of enformer output. So we'll have a total of 5313 * 2 features for training.
    human_pretrained_output = pretrained_output['human'][448:450, :]
    print(f"Printing the size of the 1st track {human_pretrained_output.size()}") #This will take  the 448th and 449th index.
    return human_pretrained_output

class EnformerDataset(Dataset):
    def __init__(self):
        referenceGenomePath = arguments["refGenomePath"]
        self.referenceGenome = pysam.FastaFile(referenceGenomePath)
        self.enformerModel = getEnformerModel()

    def __getitem__(self, index):
        dataFilePath = arguments["dataFile"]
        with h5py.File(dataFilePath, 'r+') as f:
            data = f['default']
            label = data[index][3]
            coords = (data[index][0], data[index][1], data[index][2])

        extendedCoordsForEnformer = getCoordsForEnformer(coords, self.referenceGenome)
        cfDnaFragment = getSequenceFromCoord(self.referenceGenome, extendedCoordsForEnformer)
        encodedFragment = oneHotEncodeSequence(cfDnaFragment)
        encoded_input_sequence = torch.tensor(np.float32(encodedFragment))
        enformerPrediction = getEnformerPredictions(self.enformerModel, encoded_input_sequence)
        return enformerPrediction, label

class BasicDenseLayer(nn.Module):
    def __init__(self):
        super(BasicDenseLayer, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )

def objectiveFn():
    dataset = EnformerDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    denseLayerModel = BasicDenseLayer().to(device)

    #Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(denseLayerModel.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            trainingFn(data, criterion, optimizer, denseLayerModel)


def trainingFn(data):
    data_point, label = data


# def validationFn():
#

parseInputs()
trainingFn()

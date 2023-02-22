from pathlib import Path

import numpy as np
import argparse
import pandas as pd

import h5py

arguments = {}

def parseInputs():
    parser = argparse.ArgumentParser(description=
                                     "Process bed files containing cfDNA fragment coordinates and return sequences. ")

    #Donor file path is - /Users/madhupv/Documents/uu_documents/major_research_project/data/L69-M6.donor.frag.bed
    #Recipient file path is - /Users/madhupv/Documents/uu_documents/major_research_project/data/L69-M6.recipient.frag.bed
    #SNP file path is - /Users/madhupv/Documents/uu_documents/major_research_project/data/snps.txt
    #Data store file path is - /Users/madhupv/Documents/uu_documents/major_research_project/fragmentomics/data.hdf5
    parser.add_argument('--donorFilePath', type=str, required=True)
    parser.add_argument('--recipientFilePath', type=str, required=True)
    parser.add_argument('--snpFilePath', type=str, required=True)
    parser.add_argument('--dataStoreFilePath', type=str, required=True)

    args = parser.parse_args()
    print(f"Type of args is {type(args)}")

    for path in vars(args):
        file = Path(args.__getattribute__(path))
        if not file.exists():
            print(f"The path {path} does not exist")
            raise SystemExit(1)

    arguments["donorFile"] = args.donorFilePath
    arguments["recipientFile"] = args.recipientFilePath
    arguments['snpFilePath'] = args.snpFilePath
    arguments['dataStoreFilePath'] = args.dataStoreFilePath

    print(f"Finished setting the args. The args for the file is now {arguments}")

def readAndStoreBedFiles():
    columnNames  = ["#chrom", "start", "end", "read_id", "mapq", "cigar1", "cigar2"]
    fullDonorNumpy = pd.read_csv(arguments["donorFile"],
                sep = "\t", names = columnNames, skiprows=11).to_numpy()
    fullRecipientNumpy = pd.read_csv(arguments["recipientFile"],
                sep = "\t", names = columnNames, skiprows=11).to_numpy()

    #TODO While running in HPC, remove the 0:10 condition. Right now we are only using 10 donor cfDNA fragments for training
    donorNumpy = fullDonorNumpy[0:10, 0:3]
    nrowsDonor, ncolsDonor = donorNumpy.shape
    donorLabels = np.zeros(nrowsDonor).reshape(nrowsDonor, 1)
    donorWithLabels = np.concatenate((donorNumpy, donorLabels), axis=1)

    #TODO While running in HPC, remove the 0:10 condition. Right now we are only using 10 recipient cfDNA fragments for training
    recipientNumpy = fullRecipientNumpy[0:10, 0:3]
    nrowsRecip, ncolsRecip = recipientNumpy.shape
    recipientLabels = np.ones(nrowsRecip).reshape(nrowsRecip, 1)
    recipientWithLabels = np.concatenate((recipientNumpy, recipientLabels), axis=1)

    numpyToBeStored = np.float32(donorWithLabels + recipientWithLabels)
    dataStoreFilePath = arguments["dataStoreFilePath"]

    #Store sequence data along with labels for donor and recipient file in hd5 format.
    with h5py.File(dataStoreFilePath, 'w') as h5_file:
        dset = h5_file.create_dataset("default", data=numpyToBeStored)
        h5_file.close()

    #Add in a function to join snp information to the numpy array.
    #Merge the SNP DF and the fragment DF based on where chromosome number is the same and the position lies between
    #start and end.


#This function has the attempt code to store bed file content as .npy files and use memMap to load only part of the
#data at a time.
#The resulting memMap is supposed to be the same as the numpy array we are storing, but the values are all different.
def storeAndLoadNumpyArrayAsMemmap():
    columnNames  = ["#chrom", "start", "end", "read_id", "mapq", "cigar1", "cigar2"]
    fullDonorNumpy = pd.read_csv(arguments["donorFile"],
                sep = "\t", names = columnNames, skiprows=11).to_numpy()

    #The full numpy file has a lot of string type fields which cannot be converted into int or float.
    #This is a problem because we have to give the dtype while creating a mem map. Without dtype, memMap somehow
    #populates random values.

    donorNumpy = fullDonorNumpy[0:10, 0:3]
    donorNumpyToBeStored = np.float32(donorNumpy)
    print(f"Shape of donor numpy is {donorNumpyToBeStored.shape}")
    print(f"Donor numpy :: {donorNumpyToBeStored}")
    # np.save("donorNumpy", donorNumpyToBeStored)

    donorNumpyFilePath = "/Users/madhupv/Documents/uu_documents/major_research_project/fragmentomics/donorNumpy.npy"
    donorMemMap = np.memmap(donorNumpyFilePath, mode='r+', dtype=np.float32, shape=donorNumpyToBeStored.shape)
    print(f"Shape of mam map is : {donorMemMap.shape}")
    print(f"Mapmap: :: {donorMemMap}")


    # print(np.where(newDonorNumpy[:, 2]))
    # print(pd.Series(newDonorNumpy[:, 0]).str.contains(';'))

    # recipientNumpy = pd.read_csv(arguments["recipientFile"],
    #             sep = "\t", names = columnNames, skiprows=11).to_numpy()

    # np.save("donorNumpy", donorNumpy)
    # np.save("recipientNumpy", recipientNumpy)
    #
    # donorNumpyFilePath = "/Users/madhupv/Documents/uu_documents/major_research_project/fragmentomics/donorNumpy.npy"
    #
    # donorMemMap = np.memmap(donorNumpyFilePath, mode='r', shape=donorNumpy.shape)
    # print(donorMemMap[:10, :])

#
# def getSnpInfoForFragment(chrNum, start, end):
#     columnNames = ["#chrom", "position", "base"]
#     pd.read_csv(arguments["donorFile"],
#                 sep="\t", names = columnNames)

if __name__ == "__main__":
    parseInputs()
    readAndStoreBedFiles()

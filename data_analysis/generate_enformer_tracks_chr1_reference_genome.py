import numpy as np
import torch

from enformer_pytorch import Enformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import h5py
import sys

sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')

import config
import utils

import importlib   
import time

import pysam

importlib.reload(config)
importlib.reload(utils)

arguments = {}

#File paths
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")

#Enformer output model hyperparameters
arguments["enformerBatchSize"] = config.modelHyperParameters.get("enformerBatchSize")
arguments["enformerNumberOfWorkers"] = config.modelHyperParameters.get("enformerNumberOfWorkers")

#General configs
arguments["file_sharing_strategy"] = config.modelGeneralConfigs.get("fileSharingStrategy")
arguments["enformerOutputFileCompression"] = config.modelGeneralConfigs.get("enformerOutputFileCompression")
arguments["enformerOutputFileChunkSize"] = config.modelGeneralConfigs.get("enformerOutputFileChunkSize")

device = "cuda" if torch.cuda.is_available() else "cpu"

referenceGenome = pysam.FastaFile(arguments["refGenomePath"])

class enformerSizedBitsRefGenome(Dataset):
    def __init__(self):
        self.papa = "papa"
    
    def __getitem__(self, index):
        # print(f"About to fetch the encoded fragment for {index}")
        start = 196607 * (index - 1)
        end = 196607 * index
        coords = (1, start, end)
        sequence = utils.getSequenceFromCoord(referenceGenome, coords)
        encodedFragment = utils.oneHotEncodeSequence(sequence)
        encodedFragment = torch.tensor(np.float32(encodedFragment))
        # print(f"For index {index}, coords are {coords} and shape of the encoded fragment is {encodedFragment.shape}")
        return encodedFragment
    
    def __len__(self):
        return 1268

def getEnformerPredictions(enformer_model, sequence, ntracks):
    # print("Inside get enformer prediction !!")
    with torch.no_grad():

        #For each output from enformer, get the right bin. 
        full_enformer_output = enformer_model(sequence)['human']
    
    #the enformer prediction is still in the GPU (since we sent the enformer model and one hot encoded sequence to the GPU. Numpy arrays are not supported in the GPU(GPU probably supports only tensors). So we pass the enformer prediction to CPU and convert it into a numpy array.
    #Detach is used to remove the gradients from the predictions. Gradients are similar to the weights of the model. In our case, we are only interested in the predictions and not the model training, so we remove the gradients to save space.
    full_enformer_output = full_enformer_output.detach().cpu()
    batch_size, nbins, ntracks = full_enformer_output.shape
    final_enformer_output = torch.empty(ntracks).view(1, -1)

    for i in range(batch_size):
        for j in range(nbins):
            single_bin_track = full_enformer_output[i, j, :].view(1, -1)
            final_enformer_output = torch.cat((final_enformer_output, single_bin_track), dim = 0)
    
    return final_enformer_output[1:]

#Look into how much h5py content can be compressed. Greater the compression, longer the time needed to read it again.
def storeAsH5pyFile(numSamples, numEnformerOuputSingleSample, createDataset = False, h5_file = False, 
                    enformerOutputToStore=False, currentIndex = False):
   
   dataset_name = "refGenomeEnformerOutputs"
   h5py_file_path = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/refGenomeEnformerOutput.h5py"
   
   num_h5py_samples = numSamples * 896
   #If we opening the H5PY file for the 1st time then create the dataset and return the file. 
   if createDataset: 
      print("This is the 1st time. Inside createDataset")

      if h5_file == False:
         h5_file = h5py.File(h5py_file_path, "w-")

      h5_file.create_dataset(dataset_name, (num_h5py_samples, numEnformerOuputSingleSample),
                                    compression="gzip", compression_opts=arguments["enformerOutputFileCompression"],
                                      chunks = (arguments["enformerOutputFileChunkSize"], numEnformerOuputSingleSample))
      return(h5_file)

   else:
      sizeOfOutputToStore = len(enformerOutputToStore)
      endIndex = currentIndex + sizeOfOutputToStore
      h5_file[dataset_name][(currentIndex):(endIndex),:] = enformerOutputToStore
      return endIndex

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])


#The function returns 2 numpy arrays. The 1st numpy array is the enformer output for all cfdna fragments. The second numpy array is the array of labels for all cfDNA fragments.
def storeEnformerOutput(h5_file = False):
    torch.multiprocessing.set_sharing_strategy(arguments["file_sharing_strategy"])

    nbins = 2
    ntracks = 5313

    #Set the model to eval mode first and then send it to cuda. This prevents the GPU node from running out of memory.
    enformerModel = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_checkpointing = True).eval()
    enformerModel = enformerModel.to(device)
    
    enformerInputDataset = enformerSizedBitsRefGenome()
    enformerInputDataloader = DataLoader(enformerInputDataset, batch_size=arguments["enformerBatchSize"], 
                                        num_workers=arguments["enformerNumberOfWorkers"],
                                        shuffle=True, worker_init_fn=set_worker_sharing_strategy)
    
    print(f"number of batches : {len(enformerInputDataloader)}")
    numSamples = len(enformerInputDataset)

    # #Create the datasets for storing enformer output. 
    h5_file = storeAsH5pyFile(numSamples, ntracks, True, h5_file)
    currentH5Index = 0

    for i, data in enumerate(enformerInputDataloader):
        
        #Store the filepath and the index within file to a separate CSV file. This is to ensure that we are able to locate the sample
        #so we can access the metadata(from original coordinate bed file) associated with the sample. 
        #filepath and index should have all the samples data from this batch. 
        encodedSequence = data
        
        # print(f"Printing the shape of the encoded sequence {encodedSequence.shape}", flush = True)
        # print(f"Printing the shape of label {label.shape}")
        encodedSequence = encodedSequence.to(device)
        
        #Will be of the shape [batch_size * 10626]
        enformerPrediction = getEnformerPredictions(enformerModel, encodedSequence, ntracks).detach().cpu().numpy()
    
        #The data is getting too big to load, round off enformer predictions to 3 decimal places. 
        enformerPrediction = np.around(enformerPrediction, decimals=3)
        print(f"Size of enformer output to be stored in h5py file is {enformerPrediction.shape}")
        
        """
        H5 file contents are updated every batch. To ensure that the contents are not overwritten every batch, store with indices. 
        The indices given are ascending order numbers starting from 0, this ensures that the shuffled order is maintained while storing in H5PY file. 
        """
        currentH5Index = storeAsH5pyFile(numSamples, ntracks, False, h5_file, enformerPrediction, currentH5Index)
        print(f"Finished processing batch {i}. The number of samples stored in H5PY file so far is {currentH5Index}", flush = True)

    h5_file.close()

if __name__ == '__main__':
    print(f"Start time is {time.time()}")
    storeEnformerOutput()
    print(f"End time is {time.time()}")

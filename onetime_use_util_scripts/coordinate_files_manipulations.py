import pandas as pd
import pysam

import h5py
import sys

sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')

import config
import utils

import importlib   
import os
import time
import random
import numpy as np
from itertools import chain

importlib.reload(config)
importlib.reload(utils)

arguments = {}
arguments["trainingCoordsDatasetName"] = config.datasetNames.get("trainingCoords")
arguments["validationCoordsDatasetName"] = config.datasetNames.get("validationCoords")
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")

#NOTE this scrirpt mostly does not require any config file changes. Only constants like dataset names, reference genome paths etc 
#come from the config files. Make sure to change the old_dir_path and new_dir_path variables in the main function. 

def getLabelsForData(dataNumpy, label):
    nrows, ncols = dataNumpy.shape
    if label == 0:
        return np.zeros(nrows).reshape(nrows, 1)
    if label == 1:
        return np.ones(nrows).reshape(nrows, 1)
    else:
        print(f"Invalid label for data : {label}")
        raise SystemExit(1)
    
"""
In a given dir_path, get the indices and filenames of all the samples that are too close to the edge. 
Returns training_filename_index an validation_filename_index. They are each maps of filename vs the list of indices within the filename 
that are too close to the edge. The default distance allowed from the edges is 100000. If you want to change it, its the 1st line in this function. 
"""
def getSamplesTooCloseToEdge(dir_path):
    allowed_distance_from_edges = 100000

    refGenomePath = arguments["refGenomePath"]
    refGenome = pysam.FastaFile(refGenomePath)

    chrom_len_map = {}
    for i in range(1, 23):
        chrom = str(i)
        length = refGenome.get_reference_length("chr" + chrom)
        chrom_len_map[chrom] = length

    length_x = refGenome.get_reference_length("chrX")
    length_y = refGenome.get_reference_length("chrY")
    chrom_len_map["X"] = length_x
    chrom_len_map["Y"] = length_y

    print(f"Finished getting chrom length map: {chrom_len_map}")
    training_filename_index = {}
    valid_filename_index = {}

    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        with h5py.File(filepath, 'r') as f:
            coords = f["trainingCoords"][:]
            for i in range(len(coords)):
                x, y, z = coords[i]
                chrom = x.decode('UTF-8')
                start = int(y)
                end = int(z)
                if(start < allowed_distance_from_edges or end >=  chrom_len_map[chrom] - allowed_distance_from_edges):
                    if filename not in training_filename_index:
                        training_filename_index[filename] = []
                    training_filename_index[filename].append(i)
            
            valid_coords = f["validationCoords"][:]
            for i in range(len(valid_coords)):
                x, y, z = valid_coords[i]
                chrom = x.decode('UTF-8')
                start = int(y)
                end = int(z)
                if(start < allowed_distance_from_edges or end >=  chrom_len_map[chrom] - allowed_distance_from_edges):
                    if filename not in valid_filename_index:
                        valid_filename_index[filename] = []
                    valid_filename_index[filename].append(i)
    
    print(f"Finished iterating through all the files and getting problematic indices", flush=True)
    print(f"Samples in training filename index : {len(list(chain.from_iterable(training_filename_index.values())))} and validation filename index : {len(list(chain.from_iterable(valid_filename_index.values())))}", flush=True)
    print(f"Training filenames are {list(training_filename_index.keys())}, validation filename are {list(valid_filename_index.keys())}")
    return training_filename_index, valid_filename_index

def dropEndFragmentSamples(filename, samples, end_filename_index):
    df = pd.DataFrame(data = samples, columns = ["chrom", "start", "end"])

    if filename in end_filename_index:
        df = df.drop(end_filename_index[filename])
        df.reset_index(inplace=True, drop=True)

    return df.values

"""
Takes the training_filename_index and validation_filename_index which are outputs from the previous function. 
Given a dict of filenames and indices, it copes all the contents from the old_dir_path, but leaves out the indices mentioned in the dict.
The path of the new coord dir is specified as new_dir_path
"""
def removeEndFragmentsBalanceClass(training_filename_index, valid_filename_index, old_dir_path, new_dir_path):
    print(f"Inside remove end fragments function. ")
    for filename in os.listdir(old_dir_path):
        if "recipient" in filename: continue
        print(f"Processing filename : {filename}")
        donor_path = os.path.join(old_dir_path, filename)

        recip_filename = filename.replace("donor", "recipient")
        recip_path = donor_path.replace("donor", "recipient")

        with h5py.File(donor_path, 'r') as f:
            donor_training = f["trainingCoords"][:]
            donor_validation = f["validationCoords"][:]
        
        with h5py.File(recip_path, 'r') as f:
            recip_training = f["trainingCoords"][:]
            recip_validation = f["validationCoords"][:]
        
        print(f"Sizes of all arrays BEFORE removing end fragments : {donor_training.shape, donor_validation.shape, recip_training.shape, recip_validation.shape}", flush=True)
        donor_training = dropEndFragmentSamples(filename, donor_training, training_filename_index)
        donor_validation = dropEndFragmentSamples(filename, donor_validation, valid_filename_index)
        recip_training = dropEndFragmentSamples(recip_filename, recip_training, training_filename_index)
        donor_training = dropEndFragmentSamples(recip_filename, recip_validation, valid_filename_index)
        print(f"Sizes of all arrays AFTER removing end fragments : {donor_training.shape, donor_validation.shape, recip_training.shape, recip_validation.shape}", flush=True)

        min_training_length = min(len(donor_training), len(recip_training))
        min_validation_length = min(len(donor_validation), len(recip_validation))

        train_indices = random.sample(range(0, min_training_length), min_training_length)
        validation_indices = random.sample(range(0, min_validation_length), min_validation_length)

        new_donor_training = donor_training[train_indices]
        new_donor_validation= donor_validation[validation_indices]
        new_recip_training = recip_training[train_indices]
        new_recip_validation = recip_validation[validation_indices]

        donor_training_labels = getLabelsForData(new_donor_training, 1)
        donor_validation_labels = getLabelsForData(new_donor_validation, 1)
        recip_training_labels = getLabelsForData(new_recip_training, 0)
        recip_validation_labels = getLabelsForData(new_recip_validation, 0)

        donor_file_name = filename
        recip_file_name = donor_file_name.replace("donor", "recipient")
        new_donor_path = os.path.join(new_dir_path, donor_file_name)
        new_recip_path = os.path.join(new_dir_path, recip_file_name)
        print(f"Filename: {donor_file_name} training data: {len(new_donor_training)}, training labels: {len(donor_training_labels)}, validation data: : {len(new_donor_validation)} and validation labels: {len(donor_validation_labels)}", flush=True)
        print(f"Filename: {recip_file_name} training data: {len(new_recip_training)}, training labels: {len(recip_training_labels)}, validation data: : {len(new_recip_validation)} and validation labels: {len(recip_validation_labels)}", flush=True)
        
        # with h5py.File(new_donor_path, 'w') as h5_file:
        #     h5_file.create_dataset(arguments["trainingCoordsDatasetName"], data=new_donor_training, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["trainingLabelsDatasetName"], data=donor_training_labels, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["validationCoordsDatasetName"], data=new_donor_validation, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["validationLabelsDatasetName"], data=donor_validation_labels, compression = "gzip", compression_opts=9)

        # with h5py.File(new_recip_path, 'w') as h5_file:
        #     h5_file.create_dataset(arguments["trainingCoordsDatasetName"], data=new_recip_training, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["trainingLabelsDatasetName"], data=recip_training_labels, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["validationCoordsDatasetName"], data=new_recip_validation, compression = "gzip", compression_opts=9)
        #     h5_file.create_dataset(arguments["validationLabelsDatasetName"], data=recip_validation_labels, compression = "gzip", compression_opts=9)

if __name__ == '__main__':
    print(f"Start time is {time.time()}", flush=True)
    old_dir_path = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingAndValidationCoordinateFiles"
    new_dir_path = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingAndValidationExactlyClassBalanced"
    training_filename_index, valid_filename_index = getSamplesTooCloseToEdge(old_dir_path)
    removeEndFragmentsBalanceClass(training_filename_index, valid_filename_index, old_dir_path, new_dir_path)
    print(f"End time is {time.time()}", flush=True)
import pandas as pd
import numpy as np
import os
import config
import argparse

"""
Iterates over all files in the directory specified in inputBedFileFolder in the config file and gets the count of 
samples for each patient. 
The output CSV has the following format
Sample name (Eg: L58-M6) , Number of samples in the donor file, Number of samples in the recipient file
The CSV file containing the output will get stored in the location specified by the command line argument --outputFilePath. 
"""
parser = argparse.ArgumentParser()
parser.add_argument("--outputFilePath", help = "Path to the csv file for storing script output")
args = parser.parse_args()

columnNames  = ["#chrom", "start", "end", "read_id", "mapq", "cigar1", "cigar2"]

inputbedFileFolder = config.filePaths.get("inputBedFileFolder")
directory = os.fsencode(inputbedFileFolder)
count = 0

fragment_numbers = []

for file in os.listdir(directory):
    filename = os.fsencode(file).decode("utf-8")
    if("recipient" in filename):
        rFilepath = inputbedFileFolder + "/" + filename
        donorFileName = filename.replace("recipient", "donor")
        print(f"Processing recipient file {filename} and donor file {donorFileName}")
        dFilepath =  inputbedFileFolder + "/" + donorFileName
        recipient_df = pd.read_csv(rFilepath,
            sep = "\t", names = columnNames, skiprows=11)
        donor_df = pd.read_csv(dFilepath,
            sep = "\t", names = columnNames, skiprows=11)

        num_recip = len(recipient_df)
        num_donor = len(donor_df)
        print(f"Number of recipient and donor fragments is {num_recip}, {num_donor}")
        file_part_name = filename.replace(".recipient.frag.bed.gz", "")
        fragment_numbers.append([file_part_name, num_donor, num_recip])
        count = count + 1

fragment_numbers_numpy = np.array(fragment_numbers)
sample_count_df = pd.DataFrame(fragment_numbers_numpy[1:,:], columns = ["Sample name", "Number of donor samples", "Number of recipient samples"])
print(sample_count_df.head())

sample_count_df.to_csv(args["outputFilePath"], index=False, header=True)

import pandas as pd
import numpy as np
import os
import config 
import importlib
importlib.reload(config)

columnNames  = ["#chrom", "start", "end", "read_id", "mapq", "cigar1", "cigar2"]

inputbedFileFolder = config.filePaths.get("inputBedFileFolder")
directory = os.fsencode(inputbedFileFolder)
count = 0

fragment_numbers = []
for file in os.listdir(directory):
    filename = os.fsencode(file).decode("utf-8")
    if("recipient" in filename):
        print(f"Processing file number {count}")
        rFilepath = inputbedFileFolder + "/" + filename
        donorFileName = filename.replace("recipient", "donor")
        print(f"Recipient filename is {filename} and donor file name is {donorFileName}")
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

print(f"Number of fragment numbers is {len(fragment_numbers)}")
fragment_numbers_numpy = np.array(fragment_numbers)
sample_count_df = pd.DataFrame(fragment_numbers_numpy[1:,:], columns = ["Sample name", "Number of donor samples", "Number of recipient samples"])
print(sample_count_df.head())

sample_count_df.to_csv(r'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/madhu_data/donor_recipient_sample_count.csv', index=False, header=True)

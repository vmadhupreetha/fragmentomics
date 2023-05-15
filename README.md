# fragmentomics

### config.py
Configuration file for storing file paths and model parameters. Other scripts for storing data and training models read this file for fetching configurations.

### create_coord_data.ipynb 
Creates H5PY files which have cfDNA coordinate information. 
1. Read chromosome number, start and end coordinates of cfDNA fragments from bed files. 
2. Split the data into training, validation and test subsets.
3. Store these subsets in H5PY files under separate datasets.

### create_donor_recipient_sample_table.py 
Script to create a CSV file/table with the number of samples present in donor and recipient samples of each patient.

### sequenceUtils.py
This file has utils methods for working with sequence data. It has the following functions. 
1. **getSequenceFromCoord** - Get the sequence string from fasta file using chromosome number, start and end coordinates
2. **oneHotEncodeSequence** - Convert a sequence string into one hot encoded numpy array 
3. **getLengthReferenceGenome** - get the length of given genome sequence as fasta file
4. **getCoordsForEnformer** - Given a set of coordinates (chromosome number, start and end), fetch a new set of coordinates so that the length matches enformer input length. 

### storeEnformerOutput.ipynb
This file has methods for getting enformer predictions from coordinates and creating H5PY files with the enformer predictions. It has functions for the folowing 
1. Read coordinates H5PY file. 
2. Get the one hot encoded sequence from the coordinates. 
3. Get enformer predictions for the sequence. 
4. Store enformer predictions in H5PY files. 

### trainModel.py
This file has methods to train a basic dense layer model with enformer prediction data. 
It has functions to 
1. Read enformer prediction data from H5PY files. 
2. Train a basic dense layer model with this data 
3. Run predictions for validation data 
4. Plot loss functions for training and validation data. 


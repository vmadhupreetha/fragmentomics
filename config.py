filePaths = {
    "inputBedFileFolder": "/hpc/compgen/projects/gw_cfdna/raw/external_data_haizi/unimputed",
    "snpFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/madhu_data/snps.txt",
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/coordinateFiles170000",
    "refGenomePath" : "/hpc/compgen/projects/fragclass/raw/hg19_genome/hg19_ch1-22_XYM.fa",
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/enformer_output_training_1269.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/enformer_output_validation_1269.hdf5",
    "testEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/enformer_output_test_270000.hdf5",
    "trainingMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingMetadata_270000.csv",
    "validationMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/validationMetadata_270000.csv",
    "testMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/testMetadata_270000.csv",
    "modelStateStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainedModels",
    "lossFunctionPlotDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/lossFunctionPlots"
}

dataCreationConfig = {
    "percentTest": 20, #Percentage of samples to be used as test (final testing of the model)
    "percentValidation": 20, #Percentage of non-training data to be used for validation (cross validation, hyperparameter tuning during training process)
    "numColsToExtract": 3 #cfDNA coords bed file has many columns. From the 1st column, how many columns of the bed file has useful data for the model.
}

modelHyperParameters = {
    "learningRate": 0.001, #0.1-0.0000001
    "numberEpochs": 3, #0
    "batchSize": 128, #32-264
    "numberOfWorkers": 12 #You have to ask in the srun -c 12 (max), how many batches is loading at the same time
}
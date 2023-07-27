filePaths = {
    "inputBedFileFolder": "/hpc/compgen/projects/gw_cfdna/raw/external_data_haizi/unimputed",
    "snpFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/madhu_data/snps.txt",
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/classBalancedCoordinates",
    "refGenomePath" : "/hpc/compgen/projects/fragclass/raw/hg19_genome/hg19_ch1-22_XYM.fa",
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_training_class_balanced.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_validation_class_balanced.hdf5",
    "testEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_test_class_balanced.hdf5",
    "trainingMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/training_metadata_class_balanced_all.csv",
    "validationMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/validation_metadata_class_balanced_all.csv",
    "testMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/test_metadata_class_balanced_all.csv",
    "modelStateStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainedModels",
    "lossFunctionPlotDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/lossFunctionPlots",
    "confusionMatrixPlotDirectory" : "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/confusionMatrixPlots",
    "performanceMetricsDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/performanceMetrics",
    "probabilityDistributionPlotDirecory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/probabilityDistributionPlotDirectory",
    "trainingAndValidationOutputsDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/1_mil_samples_diff_learning_rates",
    "classBalancedBedFilesDirectory" : "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/classBalancedCoordinateBedFiles"
}

testFilePaths = {
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/testCoordStoreDirectory",
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_training_270000_class_balanced.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_validation_270000_class_balanced.hdf5",
    "testEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/some_test_test_file.hdf5"
}

dataCreationConfig = {
    "percentTest": 20, #Percentage of samples to be used as test (final testing of the model)
    "percentValidation": 20, #Percentage of non-training data to be used for validation (cross validation, hyperparameter tuning during training process)
    "numColsToExtract": 3 #cfDNA coords bed file has many columns. From the 1st column, how many columns of the bed file has useful data for the model.
}

modelHyperParameters = {
    "learningRate": 0.000001, #0.1-0.0000001
    "numberEpochs": 20, #0
    "batchSize": 128, #32-264
    "numberOfWorkers": 12, #You have to ask in the srun -c 12 (max), how many batches is loading at the same time
    "threshold": 0.5
}

modelGeneralConfigs = {
    "storePlots": False,
    "modelName" : "cosine_learning_rate_0.001",
    "interchangeLabels" : True,
    "useClassWeights": False,
    "useCosineLearningFunction": False,
    "normalizeFeatures": False,
    "runWithControls": True, #Half of the positives in the training batch will be replaced with a value between 2 and 3 for the 100th feature.
    "startIndexEnformerSamplesTraining": 0,#From the training enformer file, start index from which samples should be used for training
    "endIndexEnformerSamplesTraining": 1023232, #From the training enformer file, end index until which samples should be used for training. Set this all "all" if all samples till the end should be taken
    "startIndexEnformerSamplesValidation":  0, #From the validation enformer file, start index from which samples should be used for validation
    "endIndexEnformerSamplesValidation":  "all", #From the validation enformer file, end index until which samples should be used for validation. Set this all "all" if all samples till the end should be taken
    "fileSharingStrategy": "file_system"
}
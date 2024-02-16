filePaths = {
    #All coordinates related file paths
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/OnlyTwoCoordinateFilesForTestingSmall",
    "inputBedFileFolder": "/hpc/compgen/projects/gw_cfdna/raw/external_data_haizi/unimputed",
    "snpFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/madhu_data/snps.txt",
    "refGenomePath" : "/hpc/compgen/projects/fragclass/raw/hg19_genome/hg19_ch1-22_XYM.fa",
    "patientTransplantStatusFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/latest_test_patients_transplant_status.csv",
    
    #All encoded sequence related file paths
    "trainingEncodedSequenceFilePath": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/encodedSequenceOutputs/trainig_encoded_sequence_final_half_mil.hdf5",
    "validationEncodedSequenceFilePath": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/encodedSequenceOutputs/validation_encoded_sequence_final_halfmil.hdf5",
    "testEncodedSequenceFilePath": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/encodedSequenceOutputs/test_encoded_sequence_smaller_subset.hdf5",

    #All enformer related file paths    
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/training_model_halfmil_attempt_2.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_model_halfmil_attempt_2.hdf5",

    "trainingEnformerTracksAverageFile" : "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/training_enformer_track_averages.csv",
    "validationEnformerTracksAverageFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_enformer_track_averages.csv",
    
    #All model output file paths
    "trainingAndValidationOutputsDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/final_report_combined_model",

    #Validating model paths
    "finalValidationModelStateDir": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/final_report_cnn_models/27_10_11_20_31_lr_0.00001_filters_40_80",
    "finalValidationModelPlotsDir": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/final_report_patient_level_plots",

    #Filenames where different intermediate model outputs are stored for plotting. 
    "trainingDataFile" : "trainingData.pkl",
    "validationDataFile": "validationData.pkl",
    "trainingLossLearningRateFile": "trainingLossLearningRateData.pkl",
    "validationLossLearningRateFile": "validationLossLearningRateData.pkl",
    "checkpointsFile": "modelCheckPoints",
    "restoreCheckpointModelDirName": "10_11_15_55_26_lr_0.00001_filters_80_160"
}

#These are configs that are used only in the storeEnformerOutput file. 
enformerScriptConfigs = {
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingAndValidationExactlyClassBalancedHalfMil",

    #Output for the script
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/training_model_halfmil_attempt_2.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_model_halfmil_attempt_2.hdf5",
    "testEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/test_final_model_1_mil.hdf5",
    "trainingEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/training_metadata_halfmil_attempt_2.csv",
    "validationEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/validation_metadata_halfmil_attempt_2.csv",
    "testEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/test_metadata_final_model_1_mil.csv",

    "modelInputSequenceSize": "enformer",

    #Enformer model related hyperparameters
    "enformerBatchSize": 8,
    "enformerNumberOfWorkers": 12,


    #General configs not related to model running
    "fileSharingStrategy": "file_system",
    "enformerOutputFileCompression": 8,
    "enformerOutputFileChunkSize": 200
}


#Intermediate data is stored in H5PY files. Different types of data can be stored in the same H5PY files under different datasets. 
datasetNames = {
    #Coordinate H5PY file related
    "trainingCoords": "trainingCoords",
    "validationCoords": "validationCoords",
    "testCoords": "testCoords",

    #Enformer output H5PY file related 
    "trainingEnformerOutput": "trainingEnformerOutput",
    "validationEnformerOutput": "validationEnformerOutput",
    "testEnformerOutput": "testEnformerOutput",

    #One hot encoded sequence H5PY file related. 
    "trainingEncodedSequence": "trainingEncodedSequence",
    "validationEncodedSequence": "validationEncodedSequence",
    "testEncodedSequence": "testEncodedSequence",

    #Labels
    "trainingLabels": "trainingLabels",
    "validationLabels": "validationLabels",
    "testLabels": "testLabels"
}

dataCreationConfig = {
    "percentTest": 20, #Percentage of samples to be used as test (final testing of the model)
    "percentValidation": 20, #Percentage of non-training data to be used for validation (cross validation, hyperparameter tuning during training process)
    "numColsToExtract": 3, #cfDNA coords bed file has many columns. From the 1st column, how many columns of the bed file has useful data for the model.
    "balanceClassesBeforeCreatingCoordiantes": True
}

modelHyperParameters = {
    #CNN and dense layer related hyperparameters
    "learningRate": 0.00001, 
    "useCosineLearningFunction": False, 
    "numberEpochs": 20,
    "batchSize": 128,
    "numberOfWorkers": 12, #The same number of CPU cores have to be requested during srun using -c <numWorkrs>
    "classificationThreshold": 0.5,
    "dropoutProbability": 0,
    "weightDecay": 0,

    #This will be "enformer" for generating Enformer output from the sequence. In other cases, it will be 370 (slightly longer than the longest fragment 366)
    "modelInputSequenceSize": 370, #Old model had 330, new model is 370
}

modelGeneralConfigs = {
    "modelName" : "combined_model_all_enformer_tracks",  #The output plots directory name will be based on this.   
    
    "restoreFromCheckpoint": False, #If an interrupted training process has to be started, then model will be loaded from the previous checkpoint. 

    #General configs (not hyperparameters) related to structure of model and its general configurations
    "storePlots": True,
    "interchangeLabels" : False,
    "useClassWeights": False,
    "normalizeFeatures": False,
    "usePaddingForCnn": True,
    "addLengthAsFeature": True, 

    #Patient level testing configs
    "ddCfDnaPercentageThreshold": 1,

    #configs for adding controls 
    "runWithControls": True,
    "percentageSamplesAsControls": 70,
    "percentageFeaturesAsControls": 70,
    "numSimulatedTrainingSamples": 500000,
    "numSimulatedValidationSamples": 125000,

    #Enformer output related configs
    "startIndexEnformerSamplesTraining": 0,#From the training enformer file, start index from which samples should be used for training
    "endIndexEnformerSamplesTraining": "all", #From the training enformer file, end index until which samples should be used for training. Set this to "all" if all samples till the end should be taken
    "startIndexEnformerSamplesValidation":  0, #From the validation enformer file, start index from which samples should be used for validation
    "endIndexEnformerSamplesValidation": "all", #From the validation enformer file, end index until which samples should be used for validation. Set this all "all" if all samples till the end should be taken.    
    "averageEnformerOutputBins": True, #Enformr Output from 2 bins are used for analysis. If this option is set to true, instead of treating tracks from 2 bins are separate  features, they'll be averaged to get a single feature. 
}

#--------------------------------------------- TEST CONFIGS START HERE -----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#This is used exclusively for testing purposes for small models.
testFilePaths = {
    "coordStoreDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/OnlyTwoCoordinateFilesForTestingSmall",
    "trainingEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/training_test.hdf5",
    "validationEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_test.hdf5",
    "trainingEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/training_metadata_test.csv",
    "validationEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/validation_metadata_test.csv",
    "testEnformerMetadata": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/metadataFiles/test_metadata_test.csv",
    "testEnformerOutputStoreFile": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/some_test_test_file.hdf5",
    "trainingAndValidationOutputsDirectory": "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/test_enformer"
}

testConfigs = {
    "modelName": "feature_10_sample_10_percents",
    "useCosineLearningFunction": False, 
    "learningRate": 0.00001,
    "numberEpochs": 5,
    "runWithControls": False,
    "restoreFromCheckpoint": False,
    "storePlots": False,
    "numSimulatedTrainingSamples": 10000,
    "numSimulatedValidationSamples": 2000,
    "percentageFeaturesAsControls": 10,
    "percentageSamplesAsControls": 10,
}
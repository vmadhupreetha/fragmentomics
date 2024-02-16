## Explanations of all config properties from the config.py file. 

### File and directory paths 
| S.No | Config Property                    | Description                                                                                                            |
|------|------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 1    | coordStoreDirectory               | Directory where H5PY files containing cfDNA fragment sequences are stored. Files are present in donor/recipient pairs, with one pair per patient and files already split into training, validation, and test sets. |
| 2    | inputBedFileFolder                | Directory containing bed files, present in donor/recipient pairs with each pair corresponding to one patient. Raw data for downstream processes.                                             |
| 3    | refGenomePath                     | Path where Reference genome is stored, used for extracting sequence from cfDNA coordinates.                             |
| 4    | patientTransplantStatusFile       | CSV file containing patient-level metadata such as % true donor fragments, clinical signs of rejection, etc.            |
| 5    | trainingEncodedSequenceFilePath   | Path to H5PY file storing one-hot encoded sequences of cfDNA fragments in the training set, used for training the combined model.                                             |
| 6    | validationEncodedSequenceFilePath | Path to H5PY file storing one-hot encoded sequences of cfDNA fragments in the validation set, used for training the combined model.                                           |
| 7    | testEncodedSequenceFilePath       | Path to H5PY file storing one-hot encoded sequences of cfDNA fragments in the test set, used for training the combined model.                                                   |
| 8    | trainingEnformerOutputStoreFile   | Path to H5PY file containing Enformer predictions for cfDNA fragment sequencing in the training set.                    |
| 9    | validationEnformerOutputStoreFile | Path to H5PY file containing Enformer predictions for cfDNA fragment sequencing in the validation set.                  |
| 10   | trainingEnformerTracksAverageFile | Path to CSV file with average value of each Enformer track for first 10k samples from the trainingEnformerOutputStoreFile. Used to calculate z-scores of Enformer predictions for normalization. |
| 11   | validationEnformerTracksAverageFile | Path to CSV file with average value of each Enformer track for first 10k samples from the validationEnformerOutputStoreFile. Used to calculate z-scores of Enformer predictions for normalization. |
| 12   | trainingAndValidationOutputsDirectory | Parent output directory for model training process. Contains sub-sub directories for plots and data generated after each training process.                                             |
| 13   | finalValidationModelStateDir     | Path within trainingAndValidationOutputsDirectory for model evaluation on test patient set.                           |
| 14   | finalValidationModelPlotsDir     | Directory where final evaluation plots on test patients go (e.g., patient level confusion matrix, correlation between true and predicted % donors).                              |


### File and Directory names

| S.No | Config Property                   | Description                                                                                                            |
|------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 1    | trainingDataFile                 | Name of the pickle file where training data generated from the model is stored. Replaced every training epoch, so the file always contains data only from the latest epoch. |
| 2    | validationDataFile               | Name of the pickle file where validation data generated from the model is stored. Replaced every training epoch, so the file always contains data only from the latest epoch. |
| 3    | trainingLossLearningRateFile     | Name of the pickle file where learning rates and losses for the training set are stored. Stored separately from trainingData to reduce file size. Contains data from all epochs. |
| 4    | validationLossLearningRateFile   | Name of the pickle file where learning rates and losses for the validation set are stored. Stored separately from trainingData to reduce file size. Contains data from all epochs. |
| 5    | checkpointsFile                  | Name of the file where model checkpoints are stored.                                                                  |
| 6    | restoreCheckpointModelDirName    | If restoreFromCheckpoint is true, this is the trainingAndValidationOutputsDirectory directory for the model created in the previous training process. The model will look for the checkpoints file to load model weights in this directory. |


### Dataset names 

| S.No | Config Property              | Description                                                                                                                                                                          |
|------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | trainingCoords               | Dataset name for the training set in coordinate H5PY files within coordStoreDirectory.                                                                                            |
| 2    | validationCoords             | Dataset name for the validation set in coordinate H5PY files within coordStoreDirectory.                                                                                          |
| 3    | testCoords                   | Dataset name for the test set in coordinate H5PY files within coordStoreDirectory.                                                                                                |
| 4    | trainingEnformerOutput       | Dataset name for the training set in the H5PY file with Enformer predictions configured in trainingEnformerOutputStoreFile.                                                      |
| 5    | validationEnformerOutput     | Dataset name for the validation set in the H5PY file with Enformer predictions configured in validationEnformerOutputStoreFile.                                                    |
| 6    | testEnformerOutput           | Dataset name for the test set in the H5PY file with Enformer predictions. This file was not created because the DL model trained on Enformer predictions was not used for final evaluations with test patients, so this dataset is never used. |
| 7    | trainingEncodedSequence      | Dataset name for the training set in the H5PY file with one-hot encoded sequences configured in trainingEncodedSequenceFilePath.                                                   |
| 8    | validationEncodedSequence    | Dataset name for the validation set in the H5PY file with one-hot encoded sequences configured in validationEncodedSequenceFilePath.                                                 |
| 9    | testEncodedSequence          | Dataset name for the test set in the H5PY file with one-hot encoded sequences configured in testEncodedSequenceFilePath.                                                           |
| 10   | trainingLabels               | Dataset name for training labels in all H5PY files.                                                                                                                                  |
| 11   | validationLabels             | Dataset name for validation labels in all H5PY files.                                                                                                                                |
| 12   | testLabels                   | Dataset name for test labels in all H5PY files.                                                                                                                                      |


### Enformer configs (To be used for the part of the pipeline where Enformer predictions are created from cfDNA fragment sequence and stored into h5PY files ) aka storeEnformerOutput.py script)

| S.No | Config Property                  | Description                                                                                                                                           |
|------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | coordStoreDirectory              | The path to the directory with the H5PY coordinate files containing cfDNA sequence for generating Enformer output.                                     |
| 2    | trainingEnformerOutputStoreFile  | Path to the H5PY file where predictions from the training set are to be stored.                                                                       |
| 3    | validationEnformerOutputStoreFile | Path to the H5PY file where predictions from the validation set are to be stored.                                                                   |
| 4    | testEnformerOutputStoreFile      | Path to the H5PY file where predictions from the test set are to be stored.                                                                           |
| 5    | trainingEnformerMetadata         | Path to the CSV file where metadata for the training set during generation of Enformer predictions is stored (Metadata refers to the patient file name and the index within the file for each sample - for tracing back). |
| 6    | validationEnformerMetadata      | Path to the CSV file where metadata for the validation set during generation of Enformer predictions is stored (Metadata refers to the patient file name and the index within the file for each sample - for tracing back). |
| 7    | testEnformerMetadata             | Path to the CSV file where metadata for the test set during generation of Enformer predictions is stored (Metadata refers to the patient file name and the index within the file for each sample - for tracing back). |
| 8    | modelInputSequenceSize          | The final length of the one hot encoded sequence that is used for generating Enformer predictions. Default value is “enformer” - internally using the value 196607, the standard required input size for Enformer. |
| 9    | enformerBatchSize               | Batch size for generating Enformer predictions (for how many samples predictions are generated in parallel). If batch size exceeds 8, GPU crashes.      |
| 10   | enformerNumberOfWorkers         | Number of CPU parallel threads while loading data (in dataloader). Range is 0 to 12.                                                                   |
| 11   | fileSharingStrategy             | How Enformer output file is accessed by parallel processes. Default is “file_system” to prevent errors.                                                |
| 12   | enformerOutputFileCompression   | Compression unit for Enformer output H5PY file. Higher compression results in lower space requirements but slower data loading.                           |
| 13   | enformerOutputFileChunkSize     | Number of samples which form a “chunk” in Enformer output H5PY file. Retrieval of multiple samples within a chunk is faster during a single iteration.   |


### Data pre-processing configs 

| S.No | Config Property                            | Description                                                                                                                                                                           |
|------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | percentTest                                | Percentage of samples to be allocated to the test set.                                                                                                                                |
| 2    | percentValidation                          | Percentage of non-test samples that should be allocated to the validation set.                                                                                                        |
| 3    | numColsToExtract                           | Number of columns from the bed files to extract, to be part of cfDNA coordinates for further processing. Most columns have metadata which is not relevant for training the models. |
| 4    | balanceClassesBeforeCreatingCoordinates   | Boolean. If set to true, before partitioning into training, validation, and test sets for creating H5PY coordinate files, the file (donor/recipient) with the majority samples for a patient is truncated such that the number of donor and recipient samples are the same for each patient (class balanced). |


### Model hyperparameters

| S.No | Config Property               | Description                                                                                                                                                                              |
|------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | learningRate                  | Learning rate for gradient descent. If useCosineLearningFunction is true, then this is the initial learning rate for the 1st batch.                                                     |
| 2    | useCosineLearningFunction     | Boolean. If true, the learning rate is dynamically varied during training following the cosine function.                                                                               |
| 3    | numberEpochs                  | Number of epochs (Training iterations).                                                                                                                                                 |
| 4    | batchSize                     | Batch size for training (number of samples processed before updating weights).                                                                                                          |
| 5    | numberOfWorkers               | Number of parallel CPU processes that the dataloader can use for loading data. Range from 0 to 12.                                                                                     |
| 6    | classificationThreshold       | The threshold on probabilities predicted by the model for the given sample having label 1 (donor-derived). If predicted probability crosses the threshold, the sample is predicted to be a donor. |
| 7    | dropoutProbability            | The proportion of neurons’ output to be dropped in dropout regularization.                                                                                                              |
| 8    | weightDecay                   | Multiplier for the weight term in the Loss function. Part of overfitting prevention method. Higher weightDecay results in a higher penalty for overfitting. Set to 0 for no penalty.   |
| 9    | modelInputSequenceSize        | Length of the final one-hot encoded sequence that is input into the CNN model. Individual cfDNA fragments are of varying lengths, but the CNN input should be fixed and depends on the architecture of the convolutional filters. If this value is changed, filter size, max pooling size, etc., have to be changed accordingly to ensure the output after convolutional filters is consistent with the size of the input layer of the feedforward layers. |


### General model configs 

| S.No | Config Property                         | Description                                                                                                                                                                                                                                        |
|------|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | modelName                               | A user-chosen name for the model being trained. Used to name the directory where plots generated during/after training and validation are stored.                                                                                                 |
| 2    | restoreFromCheckpoint                  | Boolean. If true, the model weights/checkpoint are updated with values from the directory configured as restoreCheckpointModelDirName in trainingAndValidationOutputsDirectory before starting the training/validation process. If false, a new model is trained from scratch. |
| 3    | storePlots                              | Boolean. If true, plots generated during/after the training/validation process are stored in trainingAndValidationOutputsDirectory in addition to being displayed.                                                                             |
| 4    | interchangeLabels                       | Boolean. Before training the models, labels are interchanged (1s are replaced with 0’s and vice versa) to fix incongruence in labels generated at the start of the project vs. at the end.                                                          |
| 5    | useClassWeights                         | Boolean. If true, class weights are used to unbias the model from the majority class before training.                                                                                                                                            |
| 6    | normalizeFeatures                       | Boolean. If true, Enformer predictions are normalized (using z-scores) to equalize the impact of all tracks.                                                                                                                                     |
| 7    | usePaddingForCnn                        | Boolean. If true, cfDNA fragment sequence is extended by padding to attain the constant CNN filter input length. If false, extension is done by overlapping with the reference genome and taking the surrounding regions until the required length is attained.                                               |
| 8    | addLengthAsFeature                      | Boolean. If true, the original length of the fragment (before extension) is used as a feature to train the models.                                                                                                                               |
| 9    | ddCfDnaPercentageThreshold             | Threshold for ddCfDNA percentage. If the model’s predicted ddCfDNA for a patient crosses this threshold, the patient is predicted as having rejection.                                                                                          |
| 10   | runWithControls                         | Boolean. If true, the models are trained on simulated data with augmented signal instead of real data from cfDNA fragments.                                                                                                                      |
| 11   | startIndexEnformerSamplesTraining       | The index in the file specified in trainingEnformerOutputStoreFile from which samples need to be considered for training.                                                                                                                        |
| 12   | endIndexEnformerSamplesTraining         | The index in the file specified in trainingEnformerOutputStoreFile until which samples need to be considered for training.                                                                                                                       |
| 13   | startIndexEnformerSamplesValidation     | The index in the file specified in validationEnformerOutputStoreFile from which samples need to be considered for training.                                                                                                                      |
| 14   | endIndexEnformerSamplesValidation       | The index in the file specified in validationEnformerOutputStoreFile until which samples need to be considered for training.                                                                                                                     |
| 15   | averageEnformerOutputBins               | Boolean. If true, Enformer predictions from the two bins corresponding to the original cfDNA fragment are averaged to get 5,313 features instead of 10,626 features.                                                                           |


### Configs for running simulations 

| S.No | Config Property                | Description                                                                                                                                                                  |
|------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | percentageSamplesAsControls    | Percentage of total samples augmented with clear signal distinguishing between positive and negatives.                                                                    |
| 2    | percentageFeaturesAsControls   | Percentage of total features augmented with clear signal distinguishing between positive and negatives.                                                                   |
| 3    | numSimulatedTrainingSamples    | Number of training samples for running simulations.                                                                                                                         |
| 4    | numSimulatedValidationSamples  | Number of validation samples for running simulations.                                                                                                                       |

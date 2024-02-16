# fragmentomics

| S.No | Config Property                      | Description                                                                                              |
|------|--------------------------------------|----------------------------------------------------------------------------------------------------------|
| 1    | coordStoreDirectory                  | Directory where H5PY files containing cfDNA fragment sequences are stored. Files are split into training, validation, and test sets.                                      |
| 2    | inputBedFileFolder                   | Directory containing bed files. Each pair corresponds to one patient and serves as raw data for downstream processes.                                                   |
| 3    | refGenomePath                        | Path where Reference genome is stored. Used for extracting sequence from cfDNA coordinates.                                                                            |
| 4    | patientTransplantStatusFile          | CSV file containing patient-level metadata like % true donor fragments, clinical signs of rejection, etc.                                                              |
| 5    | trainingEncodedSequenceFilePath      | Path to H5PY file where one hot encoded sequences of cfDNA fragments in the training set are stored. Used for training the combined model.                             |
| 6    | validationEncodedSequenceFilePath    | Path to H5PY file where one hot encoded sequences of cfDNA fragments in the validation set are stored. Used for training the combined model.                           |
| 7    | testEncodedSequenceFilePath          | Path to H5PY file where one hot encoded sequences of cfDNA fragments in the test set are stored. Used for training the combined model.                                 |
| 8    | trainingEnformerOutputStoreFile      | Path to H5PY file containing Enformer predictions for cfDNA fragment sequencing in the training set.                                                                  |
| 9    | validationEnformerOutputStoreFile    | Path to H5PY file containing Enformer predictions for cfDNA fragment sequencing in the validation set.                                                                |
| 10   | trainingEnformerTracksAverageFile    | Path to CSV file with the average value of each Enformer track for the first 10k samples from the trainingEnformerOutputStoreFile. Used to calculate z-scores of Enformer predictions for normalization. |
| 11   | validationEnformerTracksAverageFile  | Path to CSV file with the average value of each Enformer track for the first 10k samples from the validationEnformerOutputStoreFile. Used to calculate z-scores of Enformer predictions for normalization. |
| 12   | trainingAndValidationOutputsDirectory | Parent output directory for model training process. Contains subdirectories for plots and data generated after each training process.                                    |
| 13   | finalValidationModelStateDir        | Path to the trainingAndValidationOutputsDirectory for the model to be evaluated on the test patient set.                                                              |
| 14   | finalValidationModelPlotsDir        | Directory where the plots of final evaluation on test patients go (plots like patient level confusion matrix, correlation between true and predicted % donors, etc.).         |

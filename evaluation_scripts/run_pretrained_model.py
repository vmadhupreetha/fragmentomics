import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import numpy as np
import os
import h5py
import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn.functional import one_hot

import SequenceCnnModelOld
import config
import utils
import plotUtils

import importlib

importlib.reload(SequenceCnnModelOld)
importlib.reload(config)
importlib.reload(utils)
importlib.reload(plotUtils)

#Directory from which input sequence are to be read. 
# coordStoreDir = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/latest_training_validation_data/trainingAndValidationExactlyClassBalancedHalfMil_0.1"
coordStoreDir = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/subsetClassBalancedCoordinateFilesCopy"

#The directory where the model checkpoints are stored. 
modelStateDirParent = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/20_aug_cnn_trials"

#Directory where plots are to be stored. 
trainingValidationPlotsDir = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/trainingValidationPlotsAndMetrics/final_report_cnn_models"

arguments = {}
arguments["refGenomePath"] = config.filePaths.get("refGenomePath")

#Model hyperparameters 
arguments["threshold"] = config.modelHyperParameters.get("classificationThreshold")
arguments["batchSize"] = config.modelHyperParameters.get("batchSize")
arguments["learningRate"] = config.modelHyperParameters.get("learningRate")
arguments["numEpochs"] = config.modelHyperParameters.get("numberEpochs")
arguments["numWorkers"] = config.modelHyperParameters.get("numberOfWorkers")

#Model general configs
arguments["modelName"] = config.modelGeneralConfigs.get("modelName")
arguments["storePlots"] = config.modelGeneralConfigs.get("storePlots")
arguments["addLengthAsFeature"] = False
arguments["ddCfDnaPercentageThreshold"] = config.modelGeneralConfigs.get("ddCfDnaPercentageThreshold")

arguments["modelInputSequenceSize"] = 330
arguments["usePaddingForCnn"] = config.modelGeneralConfigs.get("usePaddingForCnn")

#Datasets
arguments["trainingCoordsDatasetName"] = config.datasetNames.get("trainingCoords")
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationCoordsDatasetName"] = config.datasetNames.get("validationCoords")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testCoordsDatasetName"] = config.datasetNames.get("testCoords")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

#get one hot encoded sequences from the coord store directory 
class PatientSequenceDataset(Dataset):
    def __init__(self, filename, sampleType):
        self.filename = filename
        self.sampleType = sampleType

    def __getitem__(self, index):     
        filepath = os.path.join(coordStoreDir, self.filename)
        
        with h5py.File(filepath, 'r') as f:
            coord = f[arguments[f"{self.sampleType}CoordsDatasetName"]][index]

            #Each sample should have only one label, it should be a single value instead of a numpy 1D array.The [0] is to make it a single value instead of a numpy array.
            # label = f['trainingLabels'][index][0]
            label = f[arguments[f"{self.sampleType}LabelsDatasetName"]][:][index]

            # if(arguments["interchangeLabels"] == True):
            #     labels = self.interchangeLabels(labels)

            sequenceOutputLength = arguments["modelInputSequenceSize"]
            expected_sequence_length = 196607 if sequenceOutputLength == "enformer" else sequenceOutputLength
            encoded_input_sequence, _, og_sequence_length = utils.getOneHotEncodedSequenceFromCoordinates(coord, arguments["refGenomePath"],
                                                                                         sequenceOutputLength, arguments["usePaddingForCnn"])
            #For some cases, the coordinates look fine, but the sequence fetched from the fasta file has size 0. 
            #If we pass such samples to enformer for predictions, we get Einops error, due to dimension mismatch.
            assert encoded_input_sequence.shape == (expected_sequence_length, 4), f"One of the samples did not have the right dimensions({(expected_sequence_length, 4)}). The sample index is {index}, shape is {encoded_input_sequence.shape}, filename is {self.filename} and index within the file is {index}"

        return encoded_input_sequence, label, og_sequence_length

    def __len__(self):
        filepath = os.path.join(coordStoreDir, self.filename)
        with h5py.File(filepath, 'r') as f:
            length = len(f[arguments[f"{self.sampleType}LabelsDatasetName"]][:])
        return length

def getPredictionsForAllData(model_state_dir):
    cnnModel = SequenceCnnModelOld.SequenceCnnModelOld(0).to('cuda')

    #Load the previously trained model
    modelStateDir = os.path.join(modelStateDirParent, model_state_dir)
    checkpoint_path = os.path.join(modelStateDir, "modelState")
    checkpoint = torch.load(checkpoint_path)
    cnnModel.load_state_dict(checkpoint)
    cnnModel.eval()

    criterion = nn.CrossEntropyLoss()
    modelInputLabelsToRet = {"training": [], "validation":[]}
    zeros = np.zeros(shape = (1, 2))
    modelPredictionsToRet = {"training": zeros, "validation": zeros}
    total_loss = {"training": 0, "validation": 0}
    batch_count = {"training": 0, "validation": 0}

    sampleTypes = ["training", "validation"]

    for sampleType in sampleTypes:
        for filename in os.listdir(coordStoreDir):

            dataset = PatientSequenceDataset(filename, sampleType)
            dataloader = DataLoader(dataset, batch_size=arguments["batchSize"], 
                                            num_workers=arguments["numWorkers"])
            batch_count[sampleType] += len(dataloader)

            num_batches = len(dataloader)
            store_plotting_data_interval = 3 if num_batches > 3000 else 1

            for i, data in enumerate(dataloader):
                sequence, class_labels, og_sequence_length = data
                if torch.cuda.is_available():
                    sequence = sequence.to("cuda")
                    class_labels = class_labels.to("cuda")
                
                #Reshaping to have the structure (batches, channels, sequence_length, 4)
                batches, sequence_length, one_hot_base_length = sequence.shape
                sequence = sequence.reshape(batches, 1, sequence_length, one_hot_base_length)
                class_labels = class_labels.to(torch.int64).flatten()
                og_sequence_length = og_sequence_length.reshape(len(og_sequence_length), 1).to("cuda")

                probabilityLabels = one_hot(class_labels, num_classes=2).to(torch.float32)
                modelPrediction = cnnModel(sequence, og_sequence_length, arguments["addLengthAsFeature"])

                loss = criterion(modelPrediction, probabilityLabels)
                total_loss[sampleType] += loss.item()

                if(i % store_plotting_data_interval == 0):
                    modelInputLabelsToRet[sampleType].extend(class_labels.cpu())
                    modelPredictionsToRet[sampleType] = np.row_stack([modelPredictionsToRet[sampleType], modelPrediction.detach().cpu().numpy()])

                # if(i % 10 == 0):
                #     print(f"Completed predictions for batch {i}")
            
            print(f"Finished all {sampleType} batches for filename {filename}. Storing the data now!!!")

    #----------------------------------------- START PLOTTING ------------------------------------------
    trainingPlotsData = {}
    trainingPlotsData["labels"] = modelInputLabelsToRet["training"]
    trainingPlotsData["predictions"] = modelPredictionsToRet["training"][1:, :]
    trainingPlotsData["loss"] = total_loss["training"]/batch_count["training"]
    num_labels_training = len(modelInputLabelsToRet["training"])
    shape_predictions_training = modelPredictionsToRet["training"][1:, :].shape
    loss_training = trainingPlotsData["loss"]
    print(f"Data for training: {num_labels_training}, {shape_predictions_training} and {loss_training}")

    # plotsData["loss"] = avg_loss_per_batch
    validationPlotsData = {}
    validationPlotsData["labels"] = modelInputLabelsToRet["validation"]
    validationPlotsData["predictions"] = modelPredictionsToRet["validation"][1:, :]
    validationPlotsData["loss"] = total_loss["validation"]/batch_count["validation"]
    num_labels_validation = len(modelInputLabelsToRet["validation"])
    shape_predictions_validation = modelPredictionsToRet["validation"][1:, :].shape
    loss_validation = validationPlotsData["loss"]
    print(f"Data for validation: {num_labels_validation}, {shape_predictions_validation} and {loss_validation}")

    return trainingPlotsData, validationPlotsData

if __name__ == '__main__':
    print(f"Start time is {time.time()}")           
    now = datetime.now()
    filename_extension = now.strftime("%d_%m_%H_%M_%S")

    # model_state_dirs = ["29_08_20_13_38_model_2_dropout_0.4_bigger_subset_lr_0.0001", "29_08_20_14_51_model_2_dropout_0.45_bigger_subset_lr_0.0001", "28_08_21_09_52_model_2_dropout_0.5_lr_0.0001", "28_08_21_07_47_model_2_dropout_0.7_lr_0.0001"]
    model_state_dirs = ["28_08_21_03_00_model_2_weight_decay_0.1_lr_0.0001", "28_08_17_06_46_model_2_weight_decay_0.01_lr_0.0001", "28_08_17_13_11_model_2_weight_decay_0.001_lr_0.0001", "28_08_21_04_27_model_2_weight_decay_0.0001_lr_0.0001"]

    # model_names = ["dropout_0.4", "dropout_0.45", "dropout_0.5", "dropout_0.7"]
    model_names = ["weight_decay_0.1", "weight_decay_0.01", "weight_decay_0.001", "weight_decay_0.0001"]

    for i in range(0, 4):
        model_state_dir = model_state_dirs[i]
        modelname = model_names[i]

        plots_directory_name = filename_extension + "_" + modelname
        plots_directory_path = os.path.join(trainingValidationPlotsDir, plots_directory_name)
        
        os.mkdir(plots_directory_path)

        trainingPlotsData, validationPlotsData = getPredictionsForAllData(model_state_dir)
        
        output_probabilities, class_predictions = plotUtils.getClassPredictionsAndProbsFromOutput(trainingPlotsData, validationPlotsData)
        plotUtils.storeAucAndRocCurve(output_probabilities, trainingPlotsData, validationPlotsData, plots_directory_path)
        plotUtils.storeConfusionMatrixHeatMap(trainingPlotsData, validationPlotsData, class_predictions, plots_directory_path)
    
    print(f"End time is {time.time()}")   
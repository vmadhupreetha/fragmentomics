import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import h5py
from datetime import datetime
import time
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.nn.functional import one_hot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sequenceCnnModel
import config
import utils

import importlib

importlib.reload(sequenceCnnModel)
importlib.reload(config)
importlib.reload(utils)

import seaborn as sns

arguments = {}
#File paths
arguments["testCoordsDir"] = config.filePaths.get("testCoordsDir")
arguments["patientTransplantStatusFile"] = config.filePaths.get("patientTransplantStatusFile")
arguments["modelStateDirectory"] = config.filePaths.get("modelStateDirectory")
arguments["testModelPlotsDir"] = config.filePaths.get("testModelPlotsDir")
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
arguments["addLengthAsFeature"] = config.modelGeneralConfigs.get("addLengthAsFeature")
arguments["ddCfDnaPercentageThreshold"] = config.modelGeneralConfigs.get("ddCfDnaPercentageThreshold")

arguments["modelInputSequenceSize"] = config.modelHyperParameters.get("modelInputSequenceSize")
arguments["usePaddingForCnn"] = config.modelGeneralConfigs.get("usePaddingForCnn")

#Datasets
arguments["testCoordsDatasetName"] = config.datasetNames.get("testCoords")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

arguments["checkpointsFile"] = config.filePaths.get("checkpointsFile")

print(f"Arguments in validate_model script are {arguments}")
print(f"\n\n \033[1mDid you check whether interchangeLabels, sequenceDataset is set to the right configurations ??\033[0m")

#Pick sequences from coordinate store directory. 
class PatientSequenceDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __getitem__(self, index):     
        filepath = os.path.join(arguments["testCoordsDir"], self.filename)
        
        with h5py.File(filepath, 'r') as f:
            coord = f[arguments["testCoordsDatasetName"]][index]

            #Each sample should have only one label, it should be a single value instead of a numpy 1D array.The [0] is to make it a single value instead of a numpy array.
            # label = f['trainingLabels'][index][0]
            label = f[arguments["testLabelsDatasetName"]][:][index]

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
        filepath = os.path.join(arguments["testCoordsDir"], self.filename)
        with h5py.File(filepath, 'r') as f:
            length = len(f[arguments["testLabelsDatasetName"]][:])
        return length

def getParametersDescription():
    if(arguments["modelInputType"] == "Sequence"):
        numPositives = 0
        numNegatives = 0
        labelsDatasetName = arguments["testLabelsDatasetName"]
        for filename in os.listdir(arguments["testCoordsDir"]):
            filePath = os.path.join(arguments["testCoordsDir"], filename)
            with h5py.File(filePath, 'r') as f:
                samples = f[labelsDatasetName][:]
                numPositives += (samples == 1).sum()
                numNegatives += (samples == 0).sum()

    elif(arguments["modelInputType"] == "Enformer"):
        with h5py.File(arguments["testEnformerStoreFile"], 'r') as f:
            samples = f[labelsDatasetName][:]
            numPositives = (samples == 1).sum()
            numNegatives = (samples == 0).sum()

    totalSamples = numPositives + numNegatives
    description = (f"Number of samples = {totalSamples} ({numPositives} positives and {numNegatives} negatives)")
    return description

def get_softmax(preds_array):
    preds_exp = np.exp(preds_array)
    sum_array = np.sum(preds_exp, axis = 1)
    softmax_preds = preds_exp/sum_array[:, None]
    return softmax_preds

def getClassPredictionsAndProbsFromOutput(plotsData):
    threshold = arguments["threshold"]
    predicted_labels = []
    predictions = plotsData["predictions"]

    #Apply softmax function to convert the prediction into probabilities between 0 and 1. This is used for plotting
    #the frequency of the outcomes to know how sure the model was for different data points. 
    #Model prediction is a 2D array of size (batchSize * 2). The 2 values are the probabilities for the positive and negative for each sample in the batch.
    # Whichever of the 2 labels has the highest probabilities is taken as the final predicted label of the model. 
    #If the probabilities added upto 1, this would count as taking 0.5 as the threshold for considering a class as the prediction. 
    #Iterate through all the positive probabilities predictions in the batch and extend the predictions list with all the predictions for the batch. 
    softmax_preds = np.around(get_softmax(predictions), 4)
    #TODO figure out which probability is for the positives. 
    softmax_positives = softmax_preds.transpose()[1].flatten()
    for prob in softmax_positives: 
        if prob > threshold: 
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return softmax_positives, predicted_labels

def getPatientLevelPercentDonorTranplantStatusPredictions(patient_donor_count):
    col_names = ["patient_sample_id", "num_samples", "donor_percentage", "Metadata Donor Percent", "DaysPostTransplant", "Clinical_signs_of_rejection", "patient_id"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)
    patient_level_donor_percents = {}
    true_transplant_statuses = {}
    predicted_transplant_statuses = {}

    for patient_id, num_donors in patient_donor_count.items():
        condition = (patient_labels_df['patient_sample_id'] == patient_id)
        num_samples = patient_labels_df.loc[condition, 'num_samples'].values[0]

        true_transplant_status = patient_labels_df.loc[condition, 'Clinical_signs_of_rejection'].values[0]
        true_transplant_status = 1 if true_transplant_status == "Yes" else 0
        true_transplant_statuses[patient_id] = true_transplant_status

        percent_donors = (num_donors/num_samples)*100
        patient_level_donor_percents[patient_id] = percent_donors

        if(percent_donors > arguments["ddCfDnaPercentageThreshold"]):
            predicted_transplant_statuses[patient_id] = 1
        else:
            predicted_transplant_statuses[patient_id] = 0
    
    return patient_level_donor_percents, predicted_transplant_statuses, true_transplant_statuses

def getConfusionMatrixLabels(cf_matrix):
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_matrix_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_matrix_labels = np.asarray(cf_matrix_labels).reshape(2,2)
    return cf_matrix_labels

def storeConfusionMatrixHeatMap(plotsData, predicted_labels, plotsDirectoryPath):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 8))
    heatmap_description = getParametersDescription()
    fig.text(.5, -0.1, heatmap_description, ha = 'center', fontsize=12)
    
    #Get fragment level confusion matrix
    fragment_true_labels = plotsData["labels"]
    fragment_cf_matrix = confusion_matrix(fragment_true_labels, predicted_labels)
    fragment_cf_matrix_labels = getConfusionMatrixLabels(fragment_cf_matrix)

    #Get patient level confusion matrix
    predicted_transplant_status = plotsData["predicted_transplant_status"]
    true_transplant_status = plotsData["true_transplant_status"]

    patient_cf_matrix = confusion_matrix(list(true_transplant_status.values()), list(predicted_transplant_status.values()))
    patient_cf_matrix_labels = getConfusionMatrixLabels(patient_cf_matrix)

    s1 = sns.heatmap(fragment_cf_matrix, annot=fragment_cf_matrix_labels, fmt = '', cmap="Blues", ax=ax1, annot_kws={"fontsize":12})
    s2 = sns.heatmap(patient_cf_matrix, annot=patient_cf_matrix_labels, fmt = '', cmap="Blues", ax=ax2, annot_kws={"fontsize":12})
    s1.set_xlabel("Predicted Label", fontsize=12)
    s1.set_ylabel("True Label", fontsize=12)
    s2.set_xlabel("Predicted Label", fontsize=12)
    s2.set_ylabel("True Label", fontsize=12)
    fig.subplots_adjust(hspace=0.75, wspace=0.75)

    ax1.title.set_text(f'Fragment level')
    ax2.title.set_text(f'Patient level')

    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "confusionMatrix")
        plt.savefig(plotPath, bbox_inches='tight')

    plt.show()
    plt.clf()

def storePredictedDonorsCorrelationPlot(plotsData, plotsDirectoryPath):
    col_names = ["patient_sample_id", "num_samples", "donor_percentage", "Metadata Donor Percent", "DaysPostTransplant", "Clinical_signs_of_rejection", "patient_id"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)
    donor_percentage_predictions = list(plotsData["predicted_donor_percentage"].values())
    true_donor_percentage = patient_labels_df[patient_labels_df['patient_sample_id'].isin( plotsData["predicted_donor_percentage"].keys())]["donor_percentage"]
    plt.scatter(true_donor_percentage, donor_percentage_predictions)
    plt.title(f'Correlation between predicted vs actual % donors')
    plt.xlabel('True % donors')
    plt.ylabel('Predicted %')
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "percentageDonorsCorrelationPlot")
        plt.savefig(plotPath, bbox_inches='tight')
    plt.show()
    plt.clf()

#Plot a seaborn joint plot- so create a new dataframe - with the columns - daysPostTransplant, predictedDonorpercentage, truePredictedDonorPercentage and clinical signs of rejection and patient
def storePercentDdCfDnaTransplantDaysPlot(plotsData):
    col_names = ["patient_sample_id", "num_samples", "donor_percentage", "Metadata Donor Percent", "DaysPostTransplant", "Clinical_signs_of_rejection", "patient_id"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)

    #Cope only some rows from the transplant status file to a new df - the rows for which the model predictions were obtained
    predicted_percent_donors = plotsData["predicted_donor_percentage"]
    condition = patient_labels_df['patient_sample_id'].isin(predicted_percent_donors.keys())
    to_plot_df = patient_labels_df[condition].copy()
    to_plot_df.reset_index(drop=True, inplace=True)

    #To the copied df, add the percentage donor values from the model predictions
    predicted_percent_donors_list = []
    for index, row in to_plot_df.iterrows():
        patient_sample_id = row["patient_sample_id"]
        print(f"Row patient_sample_id: {patient_sample_id}")
        predicted_percent_donors_list.append(predicted_percent_donors[row["patient_sample_id"]])
    to_plot_df["predicted_percent_donors"] = predicted_percent_donors_list
    print(f"Got the final df for plotting %dd cf-DNA vs transplant days. Printing it")
    print(to_plot_df.head(10))

    # Create the scatter plots with different color palettes
    scatter = sns.scatterplot(data=to_plot_df, x="DaysPostTransplant", y="predicted_percent_donors", hue="Clinical_signs_of_rejection")
    scatter.legend_.set_title("Clinical signs of rejection")

    sns.lineplot(data=to_plot_df, x="DaysPostTransplant", y='predicted_percent_donors', color="gray", alpha=0.2)


    # # Add gray lines connecting points with the same DaysPostTransplant value
    # for day in to_plot_df['DaysPostTransplant'].unique():
    #     predicted_data = to_plot_df[to_plot_df['DaysPostTransplant'] == day]
        
    #     # Plot gray lines
    #     plt.plot(predicted_data['DaysPostTransplant'], predicted_data['predicted_percent_donors'], color='gray', alpha=0.2)

    plt.legend()
    plt.show()

def chatGptsMakePlotsCode(plotsData):
    # Sample data
    col_names = ["patient_sample_id", "num_samples", "donor_percentage", "Metadata Donor Percent", "DaysPostTransplant", "Clinical_signs_of_rejection", "patient_id"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep="\t", names=col_names, skiprows=1)

    # Filter rows based on predicted values
    predicted_percent_donors = plotsData["predicted_donor_percentage"]
    condition = patient_labels_df['patient_sample_id'].isin(predicted_percent_donors.keys())
    to_plot_df = patient_labels_df[condition].copy()
    to_plot_df.reset_index(drop=True, inplace=True)

    # Add predicted percentage donor values from model predictions
    predicted_percent_donors_list = []
    for index, row in to_plot_df.iterrows():
        patient_sample_id = row["patient_sample_id"]
        predicted_percent_donors_list.append(predicted_percent_donors.get(patient_sample_id, None))
    to_plot_df["predicted_percent_donors"] = predicted_percent_donors_list

    # Create two DataFrames for predicted and true values
    temp_renamed_df_predicted = to_plot_df.rename(columns={'predicted_percent_donors': 'ddcfDNA %'})
    temp_renamed_df_true = to_plot_df.rename(columns={'Metadata Donor Percent': 'ddcfDNA %'})

    # Create the scatter plots with different color palettes
    sns.scatterplot(data=temp_renamed_df_predicted, x="DaysPostTransplant", y='ddcfDNA %', label="Predicted % ddcfDNA")
    sns.scatterplot(data=temp_renamed_df_true, x="DaysPostTransplant", y='ddcfDNA %', label="True % dd-cfDNA")

    # Group the data by DaysPostTransplant
    predicted_groups = temp_renamed_df_predicted.groupby("DaysPostTransplant")
    true_groups = temp_renamed_df_true.groupby("DaysPostTransplant")

    # Add gray lines connecting points in the scatter plots
    sns.lineplot(data=temp_renamed_df_predicted, x="DaysPostTransplant", y='ddcfDNA %', color="gray", alpha=0.2)
    sns.lineplot(data=temp_renamed_df_true, x="DaysPostTransplant", y='ddcfDNA %', color="gray", alpha=0.2)

    # # Plot gray lines connecting points with the same DaysPostTransplant value
    # for day, group in predicted_groups:
    #     plt.plot(group['DaysPostTransplant'], group['ddcfDNA %'], color='gray', alpha=0.2)
        
    # for day, group in true_groups:
    #     plt.plot(group['DaysPostTransplant'], group['ddcfDNA %'], color='gray', alpha=0.2)


    # Add a legend
    plt.legend()
    plt.show()

def storeAucAndRocCurve(probabilities, plotsData, plotsDirectoryPath):
    #Get AUC and TPR, FPR for training
    labels = plotsData["labels"]
    auc_score = roc_auc_score(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)

    #Get AUC, TPR, FPR for a random predictor
    random_pred_val = [0 for i in range(len(labels))]
    r_fpr, r_tpr, _ = roc_curve(labels, random_pred_val, pos_label=1)

    plt.plot(fpr, tpr, linestyle='--',color='red', label='Test model ROC curve')
    plt.plot(r_fpr, r_tpr, linestyle='--', color='black')
    plt.title(f'ROC curve plot: AUC : {auc_score}')
    plt.xlabel('False Positive Rate/FPR')
    plt.ylabel('True Positive Rate/TPR')
    plt.legend(loc='best')
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "ROC")
        plt.savefig(plotPath, bbox_inches='tight')

    plt.show()
    plt.clf()

def storeDataAndMakePlots(plotsDirectoryPath, plotsData, class_predictions, output_probabilities, modelInputType):
    plt.style.use('seaborn')
    arguments["modelInputType"] = modelInputType

    output_probabilities, class_predictions = getClassPredictionsAndProbsFromOutput(plotsData)
    
    storeAucAndRocCurve(output_probabilities, plotsData, plotsDirectoryPath)

    #Plot confusion matrix heat map for training and validation (only for last epoch)
    storeConfusionMatrixHeatMap(plotsData, class_predictions, plotsDirectoryPath)

    #Plot the correlation between predicted percentage donors and actual percentage donors for each sample 
    storePredictedDonorsCorrelationPlot(plotsData, plotsDirectoryPath)

    # Plot the predicted percentage dd-cfDNA over days of transplant for a patient. 
    
    storePercentDdCfDnaTransplantDaysPlot(plotsData)

def runModelAndGetPredictionsForPatient(patient_file_name, cnnModel, criterion):
    plotsData = {"labels": {}, "predictions": {}, "loss": {}}
    
    dataset = PatientSequenceDataset(patient_file_name)
    dataloader = DataLoader(dataset, batch_size=arguments["batchSize"], 
                                    num_workers=arguments["numWorkers"])

    num_batches = len(dataloader)
    store_plotting_data_interval = 3 if num_batches > 3000 else 1
    running_loss = 0.0

    modelPredictionsToRet = np.zeros(shape = (1, 2))
    modelInputLabelsToRet = []

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
        running_loss += loss.item()

        if(i % store_plotting_data_interval == 0):
            # modelInputDataToRet =  np.row_stack([modelInputDataToRet, sequence.detach().cpu().numpy()])
            modelInputLabelsToRet.extend(class_labels.cpu())
            modelPredictionsToRet = np.row_stack([modelPredictionsToRet, modelPrediction.detach().cpu().numpy()])
        if(i % 100 == 0):
            print(f"Completed predictions for batch {i}")
    
    print(f"Finished all batches, storing the data now !!!")
    avg_loss_per_batch = running_loss/len(dataloader)
    plotsData["loss"] = avg_loss_per_batch
    plotsData["labels"] = modelInputLabelsToRet
    plotsData["predictions"] = modelPredictionsToRet[1:, :]
    return plotsData

def getPatientLevelPredictedDonors(filename, predictions, patient_percent_donors):
    patient_id = filename.replace(".recipient.hdf5", "").replace(".donor.hdf5", "")
    predicted_donors = predictions.count(1)
    if(patient_id in patient_percent_donors):
        patient_percent_donors[patient_id] = patient_percent_donors[patient_id] + predicted_donors
    else:
        patient_percent_donors[patient_id] = predicted_donors
    return patient_percent_donors

def combinePredictionsFromAllPatients(plotsDirectoryPath):
    to_exclude_patients = ['L80-W2.donor.hdf5', 'L80-W2.recipient.hdf5', 'L68-M12.donor.hdf5', 'L68-M12.recipient.hdf5', 'L81-M2.donor.hdf5', 'L81-M2.recipient.hdf5', 'L9b-W2.donor.hdf5', 'L9b-W2.recipient.hdf5', 'L82-M1-5.donor.hdf5', 'L82-M1-5.recipient.hdf5', 'L81-D1.1.donor.hdf5', 'L81-D1.1.recipient.hdf5', 'L77-D1-3.donor.hdf5', 'L77-D1-3.recipient.hdf5', 'L33-M3.donor.hdf5', 'L33-M3.recipient.hdf5', 'L5-M13_5.donor.hdf5', 'L5-M13_5.recipient.hdf5', 'L33-M8.donor.hdf5', 'L33-M8.recipient.hdf5', 'L34-M6.donor.hdf5', 'L34-M6.recipient.hdf5', 'L16-M23.donor.hdf5', 'L16-M23.recipient.hdf5', 'L59-M6.donor.hdf5', 'L59-M6.recipient.hdf5', 'L35-W2.donor.hdf5', 'L35-W2.recipient.hdf5', 'L33-M2-5.donor.hdf5', 'L33-M2-5.recipient.hdf5', 'L2-M25.donor.hdf5', 'L2-M25.recipient.hdf5', 'L69-M6.donor.hdf5', 'L69-M6.recipient.hdf5', 'L81-M3.donor.hdf5', 'L81-M3.recipient.hdf5', 'L69-M2.donor.hdf5', 'L69-M2.recipient.hdf5', 'L30-M2.donor.hdf5', 'L30-M2.recipient.hdf5', 'L74-D2-1.donor.hdf5', 'L74-D2-1.recipient.hdf5']
    #Load state dict into the model 
    cnnModel = sequenceCnnModel.SequenceCnnModel(0).to('cuda')
    arg1 =arguments["modelStateDirectory"]
    arg2 = arguments["checkpointsFile"]
    print(f"About to get checkpoints path, the arguments are {arg1} and {arg2}")
    checkpoint_path = os.path.join(arguments["modelStateDirectory"], arguments["checkpointsFile"])
    checkpoint_dict = torch.load(checkpoint_path)
    cnnModel.load_state_dict(checkpoint_dict["model_state_dict"])
    cnnModel.eval()

    criterion = nn.CrossEntropyLoss()

    allPlotsData = {"labels": [], "predictions": [], "covered_files": []}

    all_class_predictions = []
    all_output_probabilities = []
    patient_predicted_donors = {}

    num_patients = 0
    total_loss = 0

    data_pickle_file_path = os.path.join(plotsDirectoryPath, "testPatientsData.pkl")
    percent_donors_file = os.path.join(plotsDirectoryPath, "patientPercentDonors.pkl")

    for filename in os.listdir(arguments["testCoordsDir"]):
        if filename in to_exclude_patients: continue

        print(f"Processing filename  : {filename}")
        num_patients += 1
        plotsData = runModelAndGetPredictionsForPatient(filename, cnnModel, criterion)
        loss = plotsData["loss"]
        print(f"for filename, loss is {loss}")
        total_loss += plotsData["loss"]
        if "labels" in allPlotsData:
            allPlotsData["labels"].extend(plotsData["labels"])
        else:
            allPlotsData["labels"] = plotsData["labels"]
        
        if "predictions" in allPlotsData:
            allPlotsData["predictions"].extend(plotsData["predictions"])
        else:
            allPlotsData["predictions"] = plotsData["predictions"]

        allPlotsData["covered_files"].append(filename)

        # This gets the output probability of only the positives. And the class that was predicted based on applying classification threshold to the output probability
        output_probabilities, class_predictions = getClassPredictionsAndProbsFromOutput(plotsData)
        all_class_predictions.extend(class_predictions)
        all_output_probabilities.extend(output_probabilities)

        #Store data at the end of every file
        #We need all_plots data to get appended to everytime. So no other option. 
        with open(data_pickle_file_path, "wb") as f:
            pickle.dump(allPlotsData, f)

        #Patient level predicted donors is a map of patient id vs number of predicted donors for the patient 
        patient_predicted_donors = getPatientLevelPredictedDonors(filename, class_predictions, patient_predicted_donors)
        with open(percent_donors_file, "wb") as f:
            pickle.dump(patient_predicted_donors, f)

    print(f"After processing all patients, the patient level donor predictions is {patient_predicted_donors}")

    #Code to retrieve plotsData and percent_donors from pickle files 
    with open(data_pickle_file_path, "rb") as f:
        allPlotsData = pickle.load(f)
    
    with open(percent_donors_file, 'rb') as f:
        patient_predicted_donors = pickle.load(f)
        
    patient_donor_percent, predicted_transplant_status, true_transplant_status = getPatientLevelPercentDonorTranplantStatusPredictions(patient_predicted_donors)
    allPlotsData["loss"] = total_loss/num_patients
    allPlotsData["predicted_donor_count"] = patient_predicted_donors
    allPlotsData["predicted_donor_percentage"] = patient_donor_percent
    allPlotsData["predicted_transplant_status"] = predicted_transplant_status
    allPlotsData["true_transplant_status"] = true_transplant_status
    print(f"After getting predictions, the plotsData values are {patient_predicted_donors}, {patient_donor_percent}, {predicted_transplant_status}, {true_transplant_status}")
    
    storeDataAndMakePlots(plotsDirectoryPath, allPlotsData, all_output_probabilities, all_class_predictions, modelInputType="Sequence")
if __name__ == '__main__':
    print(f"Start time is {time.time()}")
    
    now = datetime.now()
    filename_extension = now.strftime("%d_%m_%H_%M_%S")
    # modelname = arguments["modelName"]
    modelname = "all_test_patients_part_2"
    plotsDirectoryName = filename_extension + "_" + str(modelname)
    plotsDirectoryPath = os.path.join(arguments["testModelPlotsDir"], plotsDirectoryName)
    os.mkdir(plotsDirectoryPath)

    combinePredictionsFromAllPatients(plotsDirectoryPath)   
    print(f"End time is {time.time()}")   
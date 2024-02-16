import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import h5py
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sequenceCnnModel
import sequenceDataset
import config
import plotUtils
import importlib

importlib.reload(sequenceCnnModel)
importlib.reload(sequenceDataset)
importlib.reload(config)
importlib.reload(plotUtils)

import seaborn as sns

arguments = {}
#File paths
arguments["trainingAndValidationOutputsDirectory"] = config.filePaths.get("trainingAndValidationOutputsDirectoryCnn")
arguments["coordStoreDirectory"] = config.filePaths.get("coordStoreDirectory")
arguments["patientTransplantStatusFile"] = config.filePaths.get("patientTransplantStatusFile")
arguments["modelStateDictPath"] = config.filePaths.get("modelStateDictPath")

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

#Datasets
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

print(f"Arguments in validate_model script are {arguments}")

def getParametersDescription():
    if(arguments["modelInputType"] == "Sequence"):
        numPositives = 0
        numNegatives = 0
        labelsDatasetName = arguments["testLabelsDatasetName"]
        for filename in os.listdir(arguments["coordStoreDirectory"]):
            filePath = os.path.join(arguments["coordStoreDirectory"], filename)
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

def getConfusionMatrixLabels(cf_matrix):
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_matrix_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_matrix_labels = np.asarray(cf_matrix_labels).reshape(2,2)
    return cf_matrix_labels

def getPatientLevelTransplantPredictions(predicted_labels):
    col_names = ["patient_sample_id", "num_samples", "percentage_donors", "transplant_status"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)
    patient_transplant_status_predictions = []
    percentage_donors_predictions = []
    start_sample_index = 0
    end_sample_index = 0

    for i, row in patient_labels_df.iterrows():
        num_samples = row["num_samples"] 
        end_sample_index = end_sample_index + num_samples
        predicted_num_donors = predicted_labels[start_sample_index: end_sample_index].count(1)
        percentage_donors = (predicted_num_donors/num_samples) * 100
        percentage_donors_predictions.append(percentage_donors)

        #If % donor fragments for a patient exceeds threshold, set the transplant status as 1. 
        if(percentage_donors > arguments["ddCfDnaPercentageThreshold"]):
            patient_transplant_status_predictions.append(1)
        else:
            patient_transplant_status_predictions.append(0)

        start_sample_index = end_sample_index

    print(f"In get patient level transplant status, the final end index is {end_sample_index} and length of predicted labels is {len(predicted_labels)}")
    return percentage_donors_predictions, patient_transplant_status_predictions
    

def getPatientLevelCfMatrixAndLabels(predicted_labels):
    col_names = ["patient_sample_id", "num_samples", "percentage_donors", "transplant_status"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)
    _, patient_transplant_status_predcictions = getPatientLevelTransplantPredictions(predicted_labels)
    cf_matrix = confusion_matrix(patient_labels_df["transplant_status"], patient_transplant_status_predcictions)
    cf_matrix_labels = getConfusionMatrixLabels(cf_matrix)
    return cf_matrix, cf_matrix_labels

def getfragmentLevelCfMatrixAndLabels(data, predicted_labels):
    true_labels = data["labels"]
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    cf_matrix_labels = getConfusionMatrixLabels(cf_matrix)
    return cf_matrix, cf_matrix_labels

def storeConfusionMatrixHeatMap(plotsData, predicted_labels, plotsDirectoryPath):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 8))
    heatmap_description = getParametersDescription()
    fig.text(.5, -0.1, heatmap_description, ha = 'center', fontsize=12)
    
    fragment_cf_matrix, fragment_cf_matrix_labels = getfragmentLevelCfMatrixAndLabels(plotsData, predicted_labels)
    patient_cf_matrix, patient_cf_matrix_labels = getPatientLevelCfMatrixAndLabels(predicted_labels)

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


def storePredictedDonorsCorrelationPlot(predicted_labels, plotsDirectoryPath):
    col_names = ["patient_sample_id", "num_samples", "percentage_donors", "transplant_status"]
    patient_labels_df = pd.read_csv(arguments["patientTransplantStatusFile"], sep = "\t", names= col_names, skiprows=1)
    donor_percentage_predictions, _ = getPatientLevelTransplantPredictions(predicted_labels)
    plt.scatter(patient_labels_df["percentage_donors"], donor_percentage_predictions)
    plt.title(f'Correlation between predicted vs actual % donors')
    plt.xlabel('True % donors')
    plt.ylabel('Predicted %')
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "percentageDonorsCorrelationPlot")
        plt.savefig(plotPath, bbox_inches='tight')
    plt.show()
    plt.clf()

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

def makePlots(plotsData, modelInputType):
    now = datetime.now()
    filename_extension = now.strftime("%d_%m_%H_%M_%S")
    plotsDirectoryName = filename_extension + "_" + str(arguments["modelName"])
    plotsDirectoryPath = os.path.join(arguments["trainingAndValidationOutputsDirectory"], plotsDirectoryName)
    if(arguments["storePlots"]):
        os.mkdir(plotsDirectoryPath)

    plt.style.use('seaborn')
    arguments["modelInputType"] = modelInputType

    #Get the softmax probabilities and the predicted class labels (using threshold) from model output 
    output_probabilities, class_predictions = getClassPredictionsAndProbsFromOutput(plotsData)
    print(f"Output probabilities : {len(output_probabilities)} and class preds: {len(class_predictions)}")
    storeAucAndRocCurve(output_probabilities, plotsData, plotsDirectoryPath)

    #Plot confusion matrix heat map for training and validation (only for last epoch)
    storeConfusionMatrixHeatMap(plotsData, class_predictions, plotsDirectoryPath)

    #Plot the correlation between predicted percentage donors and actual percentage donors for each sample 
    storePredictedDonorsCorrelationPlot(class_predictions, plotsDirectoryPath)

def runModelAndGetPredictions():
    plotsData = {"labels": {}, "predictions": {}}
    
    #Load state dict into the model 
    cnnModel = sequenceCnnModel.SequenceCnnModel(0).to('cuda')
    cnnModel.load_state_dict(torch.load(arguments["modelStateDictPath"]))
    cnnModel.eval()
    
    dataset = sequenceDataset.SequenceDataset("test")
    dataloader = DataLoader(dataset, batch_size=arguments["batchSize"], 
                                    num_workers=arguments["numWorkers"])

    modelPredictionsToRet = np.zeros(shape = (1, 2))
    modelInputLabelsToRet = []

    for i, data in enumerate(dataloader):
        sequence, class_labels, _, _, _, og_sequence_length = data
        if torch.cuda.is_available():
            sequence = sequence.to("cuda")
            class_labels = class_labels.to("cuda")
        
        #Reshaping to have the structure (batches, channels, sequence_length, 4)
        batches, sequence_length, one_hot_base_length = sequence.shape
        sequence = sequence.reshape(batches, 1, sequence_length, one_hot_base_length)
        class_labels = class_labels.to(torch.int64).flatten()
        og_sequence_length = og_sequence_length.reshape(len(og_sequence_length), 1).to("cuda")

        modelPrediction = cnnModel(sequence, og_sequence_length, arguments["addLengthAsFeature"])

        modelInputLabelsToRet.extend(class_labels.cpu())
        modelPredictionsToRet = np.row_stack([modelPredictionsToRet, modelPrediction.detach().cpu().numpy()])

        if(i % 100 == 0):
            print(f"Completed predictions for batch {i}")
    
    print(f"Finished all batches, storing the data now !!!")
    plotsData["labels"] = modelInputLabelsToRet
    plotsData["predictions"] = modelPredictionsToRet[1:, :]

    print(f"Completed training and validation. Making plots")
    makePlots(plotsData, modelInputType="Sequence")

if __name__ == '__main__':
    print(f"Start time is {time.time()}")
    runModelAndGetPredictions()   
    print(f"End time is {time.time()}")   
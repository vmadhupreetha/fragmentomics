'''
This file contains all the methods for generating plots to visualize data during and after the training process. It
also includes methods for checkpointing model state (to be called after every training epoch) and restoring the model
from checkpoint.
Contains the following plots
1. Loss Function plot
2. Receiver Operating Characteristics (ROC) curve
3. Confusion matrix for class predictions.
4. Probability distribution plot for all predictions combined.
5. Probability distribution plots for false positives, false negatives, true positives and true negatives.
6. Precision recall curve
7. Learning rate curve (over training epochs)

If storePlots config is true, in addition to displaying plots, they are also saved to plotsDirectory.

The data for plotting is passed around in the form of a dict containing predictions of the model, true labels etc.

This is how the training data looks like. They are dict type objects with keys labels, loss, learningRates and inputData.
These guys are inturn dicts with the epoch values (in int) as keys. The data type for each epoch for these types of data are as follows
Labels[epoch_number] is a list of 0's and 1's of the labels in training order for that epoch
Predictions[epoch_number] is a numpy array of size [sample_size * 2] which are the model output probalities for positive and negative class for all samples
Loss[epoch_number] is a float value which is the cross entropy loss for that epoch (averaged over all batches)

Additionally, the trainingData also has predictions from the last epoch - which are probabilities (sort of) predicted by
the model for a given sample being donor-derived (for all samples).
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from itertools import chain

import config 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import torch 
import h5py

import seaborn as sns

arguments = {}

#File paths
arguments["trainingAndValidationOutputsDirectory"] = config.filePaths.get("trainingAndValidationOutputsDirectory")
arguments["coordStoreDirectory"] = config.filePaths.get("coordStoreDirectory")
arguments["trainingEnformerStoreFile"] = config.filePaths.get("trainingEnformerOutputStoreFile")
arguments["validationEnformerStoreFile"] = config.filePaths.get("validationEnformerOutputStoreFile")

arguments["trainingDataFile"] = config.filePaths.get("trainingDataFile")
arguments["validationDataFile"] = config.filePaths.get("validationDataFile")
arguments["trainingLossLearningRateFile"] = config.filePaths.get("trainingLossLearningRateFile")
arguments["validationLossLearningRateFile"] = config.filePaths.get("validationLossLearningRateFile")
arguments["checkpointsFile"] = config.filePaths.get("checkpointsFile")

#Model hyper parameters that are used for plotting
arguments["threshold"] = config.modelHyperParameters.get("classificationThreshold")
arguments["batchSize"] = config.modelHyperParameters.get("batchSize")
arguments["learningRate"] = config.modelHyperParameters.get("learningRate")
arguments["numEpochs"] = config.modelHyperParameters.get("numberEpochs")
arguments["useCosineLearningFunction"] = config.modelHyperParameters.get("useCosineLearningFunction")

#Plotting related general configs
arguments["storePlots"] = config.modelGeneralConfigs.get("storePlots")

#Datasets
arguments["trainingLabelsDatasetName"] = config.datasetNames.get("trainingLabels")
arguments["validationLabelsDatasetName"] = config.datasetNames.get("validationLabels")
arguments["testLabelsDatasetName"] = config.datasetNames.get("testLabels")

#!!! Test configs... set them right once the plotting is done !!!!!
# arguments["trainingAndValidationOutputsDirectory"] = config.testFilePaths.get("trainingAndValidationOutputsDirectory")
# arguments["coordStoreDirectory"] = config.testFilePaths.get("coordStoreDirectory")
# arguments["trainingEnformerStoreFile"] = config.testFilePaths.get("trainingEnformerOutputStoreFile")
# arguments["validationEnformerStoreFile"] = config.testFilePaths.get("validationEnformerOutputStoreFile")

# arguments["learningRate"] = config.testConfigs.get("learningRate")
# arguments["numEpochs"] = config.testConfigs.get("numberEpochs")
# arguments["useCosineLearningFunction"] = config.testConfigs.get("useCosineLearningFunction")

# arguments["storePlots"] = config.testConfigs.get("storePlots")

print(f"Arguments in plotUtils are {arguments}")

"""
Pickle dump the training and validation data to be used for plotting. The reason behind this is - if at some point, the model
training encounters error, then the data generated up until that point could still be used for plotting at the end, without
having to re-train from epoch 1. 

NOTE - plotsData only has data from the latest epoch. so it is important to append to the pickle file using "ab" mode. 
"""
def savePlotsData(sampleType, data, loss, learning_rates, plotsDirectoryPath):

    #Save model predictions, labels and input data 
    data_pickle_file_path = os.path.join(plotsDirectoryPath, arguments[f"{sampleType}DataFile"])
    with open(data_pickle_file_path, "wb") as f:
        pickle.dump(data, f)
    
    #Save loss and learning rate. But for these 2, append to the existing loss and learning rates because we construct plots for all epochs. 
    loss_learning_rate_file_path = os.path.join(plotsDirectoryPath, arguments[f"{sampleType}LossLearningRateFile"])
    current_epoch = data["epoch"]

    if(current_epoch == 1):
        loss_learning_rate_data = {}
        loss_learning_rate_data["loss"] = [loss]
        loss_learning_rate_data["learningRates"] = learning_rates
    else:
        with open(loss_learning_rate_file_path, "rb") as file:
            loss_learning_rate_data = pickle.load(file)
            all_losses = loss_learning_rate_data["loss"]
            all_losses.append(loss)
            loss_learning_rate_data["loss"] = all_losses

            if(sampleType == "training"):
                all_learning_rates = loss_learning_rate_data["learningRates"]
                all_learning_rates.extend(learning_rates)
                loss_learning_rate_data["learningRates"] = all_learning_rates 
    
    with open(loss_learning_rate_file_path, "wb") as file:
        pickle.dump(loss_learning_rate_data, file)

'''
Save checkpoint (weights etc) of the model. This function is called from train model scripts at the end of every training epoch
Safe guard to ensure current state is stored incase training processes errors out. 
'''
def saveModelCheckpoint(epoch, model, optimizer, plotsDirPath):
    print(f"Inside save model check point")
    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    check_point_path = os.path.join(plotsDirPath, arguments["checkpointsFile"])
    torch.save(checkpoint, check_point_path)

'''
The parent plotting function that calls child functions to generate various plots. Called from model training scripts
at the end of the training process

Inputs 
1. plotsDirectoryPath - directory where the plots should be saved 
2. modelInputType - "Enformer" or "Seqence" - only relevant for annotating the plots. 
'''
def storePlots(plotsDirectoryPath, modelInputType):
    plt.style.use('seaborn')
    arguments["modelInputType"] = modelInputType

    #Load the predictions, labels and input data data pickle files. 
    trainingPickleFilePath = os.path.join(plotsDirectoryPath, arguments["trainingDataFile"])
    with open(trainingPickleFilePath, 'rb') as file:
        trainingData = pickle.load(file)

    validationPickleFilePath = os.path.join(plotsDirectoryPath, arguments["validationDataFile"])
    with open(validationPickleFilePath, 'rb') as file:
        validationData = pickle.load(file)

    #Load the loss and the learning rates and add them to the data file
    training_loss_learning_rate_pickle_file = os.path.join(plotsDirectoryPath, arguments[f"trainingLossLearningRateFile"])
    with open(training_loss_learning_rate_pickle_file, 'rb') as f:
        trainingLossLearningRate = pickle.load(f)
    
    trainingData["loss"] = trainingLossLearningRate["loss"]
    trainingData["learningRates"] = list(chain.from_iterable(trainingLossLearningRate["learningRates"])) #Learning rates is a list of lists. Flatten it

    validation_loss_learning_rate_pickle_file = os.path.join(plotsDirectoryPath, arguments[f"validationLossLearningRateFile"])
    with open(validation_loss_learning_rate_pickle_file, 'rb') as f:
        validationLossLearningRate = pickle.load(f)
    
    validationData["loss"] = validationLossLearningRate["loss"]
    validationData["learningRates"] = validationLossLearningRate["learningRates"]
    
    #Violin plot of model input data values for positives and negatives for the 1st 25 features for training and validation
    #makePositiveNegativeDistributionPlot(trainingData, validationData)

    #Loss function plot for training and validation 
    storeLossFunctionPlot(trainingData, validationData, plotsDirectoryPath)
    
    #Get the softmax probabilities and the predicted class labels (using threshold) from model output 
    output_probabilities, class_predictions = getClassPredictionsAndProbsFromOutput(trainingData, validationData)

    storeAucAndRocCurve(output_probabilities, trainingData, validationData, plotsDirectoryPath)

    #Plot confusion matrix heat map for training and validation (only for last epoch)
    storeConfusionMatrixHeatMap(trainingData, validationData, class_predictions, plotsDirectoryPath)

    """
    Plot the predicted probabilities distribution for all samples. Only the probabilities for output being a positive are plotted
    Note - this does not mean only positive predictions. If this probability is >threshold, final prediction is positive otherwise it is negative)
    """
    storeProbabilityFrequencyPlot(output_probabilities, plotsDirectoryPath)

    """
    Plot the probability distributions of output being a positive (only for last epoch)
    Same as the previous plot, but segregated into true positives, false positives, true negatives and false negatives. 
    """
    confsionMatrixLevelProbDistribtuionPlot(trainingData, validationData, output_probabilities, class_predictions, plotsDirectoryPath)

    #Plot learning rate parameter of the model over the training iterations. 
    if(arguments["useCosineLearningFunction"]):
        plotLearningRate(trainingData, plotsDirectoryPath)

'''
Returns the text description for all plots containing info like number of samples, training epochs and other hyperparameters 
used for training. 
'''
def getParametersDescription():
    # # The model type is to know which files to read to get the total sample count 
    # # In case of the combined model, we need to read the enformer output file or the encoded sequence file. So type is passed as "enformer"
    # trainingLabelsDatasetName = arguments["trainingLabelsDatasetName"]
    # validationLabelsDatasetName = arguments["validationLabelsDatasetName"]
    
    # if(arguments["modelInputType"] == "Sequence"):
    #     numTrainingPositives = 0
    #     numTrainingNegatives = 0
    #     numValidationPositives = 0
    #     numValidationNegatives = 0
        
    #     for filename in os.listdir(arguments["coordStoreDirectory"]):
    #         filePath = os.path.join(arguments["coordStoreDirectory"], filename)
    #         with h5py.File(filePath, 'r') as f:

    #             trainingSamples = f[trainingLabelsDatasetName][:]
    #             numTrainingPositives += (trainingSamples == 1).sum()
    #             numTrainingNegatives += (trainingSamples == 0).sum()

    #             validationSamples = f[validationLabelsDatasetName][:]
    #             numValidationPositives += (validationSamples == 1).sum()
    #             numValidationNegatives += (validationSamples == 0).sum()

    # elif(arguments["modelInputType"] == "Enformer"):
    #     with h5py.File(arguments["trainingEnformerStoreFile"], 'r') as f:
    #         trainingSamples = f[trainingLabelsDatasetName][:]
    #         numTrainingPositives = (trainingSamples == 1).sum()
    #         numTrainingNegatives = (trainingSamples == 0).sum()
        
    #     with h5py.File(arguments["validationEnformerStoreFile"], 'r') as f:
    #         validationSamples = f[validationLabelsDatasetName][:]
    #         numValidationPositives = (validationSamples == 1).sum()
    #         numValidationNegatives = (validationSamples == 0).sum()

    # numTraining = numTrainingPositives + numTrainingNegatives
    # numValidation = numValidationPositives + numValidationNegatives

    # learningRate = arguments["learningRate"]

    # description = (f"learning rate = {learningRate},\n"
    #                + f"number of training samples = {numTraining} ({numTrainingPositives} positives and {numTrainingNegatives} negatives),\n"
    #                 + f"number of validation samples = {numValidation} ({numValidationPositives} positives and {numValidationNegatives} negatives)")
    
    # return description
    empty_string = " "
    return empty_string

"""
Make a violin plot of the values distributions for all positives and negatives for each feature for a given set of 
samples. The purpose of this plot is to check if the controls are working. The controls should fall somewhat outside
the normal range for positives and negatives. 
"""
def makePositiveNegativeDistributionPlot(trainingData, validationData):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (12,8)) 
    fig.tight_layout()
    
    for i, data in enumerate([trainingData, validationData]):
        #Plot the distribution curve only for the 1st epoch
        classLabels = data["labels"][1]
        inputData = data["inputData"][1]
        _, numFeatures = inputData.shape
        violin_plot_positives = np.zeros(shape = (1, numFeatures))
        violin_plot_negatives = np.zeros(shape = (1, numFeatures))

        for j, singleSample in enumerate(inputData):
            if classLabels[j] == 1:
                violin_plot_positives = np.row_stack([violin_plot_positives, singleSample])
            else:
                violin_plot_negatives = np.row_stack([violin_plot_negatives, singleSample])
        
        #plotting only for the 1st 25 features
        column_values = [str(j) for j in range(0, 25)]
        positives_df = pd.DataFrame(data = violin_plot_positives[1:, :], columns = column_values)
        sns.violinplot(data = positives_df, ax = ax[i][0],palette="Greens")
        negatives_df = pd.DataFrame(data = violin_plot_negatives[1:, :], columns = column_values)
        sns.violinplot(data = negatives_df, ax = ax[i][1], palette="Reds")

        plt.xlabel("Features")
        plt.xlabel("Distributions")
        ax[0][0].set_title("Training positives")
        ax[0][1].set_title("Training negatives")
        ax[1][0].set_title("Validation positives")
        ax[1][1].set_title("Validation negatives")

    plt.show()
    plt.clf()

'''
Inputs 
1. trainingData - python dict (refer to the file description for more details on training data) 
2. validationData - python dict (refer to the file description for more details on training data) 
3. plotsDirectoryPath - directory where the plots should be saved. 

Creates a plot of Cross entropy loss over training epochs, for both training and validation data. 
'''
def storeLossFunctionPlot(trainingData, validationData, plotsDirectoryPath):
    fig = plt.figure()
    labels = ["Training", "Validation"]

    #Add border to the plots
    ax = plt.gca()
    ax = fig.add_axes((0, 0, 1, 1))

    #this will have only 2 iterations - one for the training data and the other for the validation data 
    for i, loss_list in enumerate([trainingData["loss"], validationData["loss"]]):
        epochs = range(1, len(loss_list) + 1)
        ax.plot(epochs, loss_list, '-.', label = labels[i]) 

    # plt.xticks(range(1, 21, 2))
    # y_ticks_float = [x/100 for x in range(60, 70, 1)]
    # plt.yticks(y_ticks_float)
    
    plot_description = getParametersDescription()
    fig.text(.5, -0.3, plot_description, ha = 'center')
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(loc="upper right", fontsize = 16)
    plt.title("Training and Validation Cross entropy loss over epochs.")

    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "lossFunctionPlot")
        plt.savefig(plotPath, bbox_inches='tight')
    plt.show()
    plt.clf()

'''
Outputs 
Both the outputs are dicts with "training" and "validation" as keys. Each value is a list of length = number of samples
in that particular set (training or validation). Description of this list for the 2 outputs is as follows.
1. output_probabilities - List of probabilities predicted by the model, where each value represants the predicted probability
    for a sample belonging to the donor class. The model output is converted to probability (val between 0 and 1) by 
    applying a softmax function. 
2. predicted_labels - The list of labels (0 or 1), where each val in the list is the label assigned to a single sample 
    based on the probability (of being donor) predicted by the model. Assigning labels are done using a threshold. 
'''
def getClassPredictionsAndProbsFromOutput(trainingData, validationData):
    threshold = arguments["threshold"]
    output_probabilities = {"training": [], "validation": []}
    predicted_labels = {"training": [], "validation": []}
    sampleTypes = ["training", "validation"]
    #Apply softmax function to convert the prediction into probabilities between 0 and 1. This is used for plotting
    #the frequency of the outcomes to know how sure the model was for different data points. 

    #Model prediction is a 2D array of size (batchSize * 2). The 2 values are the probabilities for the positive and negative for each sample in the batch.
    # Whichever of the 2 labels has the highest probabilities is taken as the final predicted label of the model. 
    #If the probabilities added upto 1, this would count as taking 0.5 as the threshold for considering a class as the prediction. 
    #Iterate through all the positive probabilities predictions in the batch and extend the predictions list with all the predictions for the batch. 

    #this loop has only 2 iterations - for the training and the validation samples.
    for i, predictions in enumerate([trainingData["predictions"], validationData["predictions"]]):
        softmax_preds = np.around(get_softmax(predictions), 4)   #all the predictions (in terms of probabilities) for a batch 
        softmax_positives = softmax_preds.transpose()[1].flatten()   #all probabilities for being positive for a batch
        output_probabilities[sampleTypes[i]] = softmax_positives  
        for prob in softmax_positives: 
            if prob > threshold: 
                predicted_labels[sampleTypes[i]].append(1)
            else:
                predicted_labels[sampleTypes[i]].append(0)

    return output_probabilities, predicted_labels

'''
Inputs - 
1. probabilities - The output_probabilities from the previous function
2. trainingData - python dict (refer to the file description for more details on training data) 
3. validationData - python dict (refer to the file description for more details on training data) 
4. plotsDirectoryPath - directory where the plots should be saved. 

Plots the training and validation ROC curve. Also displays the area under the curve (AUC) for training and validation data. 
'''
def storeAucAndRocCurve(probabilities, trainingData, validationData, plotsDirectoryPath):
    training_probs = probabilities["training"]
    validation_probs = probabilities["validation"]

    #Get AUC and TPR, FPR for training
    training_labels = trainingData["labels"]
    auc_score_training = roc_auc_score(training_labels, training_probs)
    auc_score_training = "{:.3f}".format(auc_score_training)
    train_fpr, train_tpr, _ = roc_curve(training_labels, training_probs, pos_label=1)

    #Get AUC, TPR and FPR for validation
    validation_labels = validationData["labels"]
    auc_score_validation = roc_auc_score(validation_labels, validation_probs)
    auc_score_validation = "{:.3f}".format(auc_score_validation)
    valid_fpr, valid_tpr, _ = roc_curve(validation_labels, validation_probs, pos_label=1)

    # Get AUC, TPR, FPR for a random predictor
    # random_pred_val = [0 for i in range(len(validation_labels))]
    # r_fpr, r_tpr, _ = roc_curve(validation_labels, random_pred_val, pos_label=1)

    plt.plot(train_fpr, train_tpr, linestyle='--',color='red', label='Training ROC curve')
    plt.plot(valid_fpr, valid_tpr, linestyle='--',color='blue', label='Validation ROC curve')
    # plt.plot(r_fpr, r_tpr, linestyle='--', color='black')
    plt.title(f'ROC Curve (Training AUC : {auc_score_training}, Validation AUC : {auc_score_validation})')
    plt.xlabel('False Positive Rate/FPR')
    plt.ylabel('True Positive Rate/TPR')
    plt.legend(loc='best')
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "ROC")
        plt.savefig(plotPath, bbox_inches='tight')

    plt.show()
    plt.clf()

'''
Inputs -
1. output_probabilities - The output_probabilities from this function - getClassPredictionsAndProbsFromOutput
2. plotsDirectoryPath - directory where the plots should be saved. 

Generates the probability distribution plot, where x axis has probabilities ranging from 0 to 1 and y axis has the distribution. 
Here, probabilities refer to the softmax probabilities generated from the prediction of a model denoting the likelihood of 
a sample being donor-derived (label 1)
NOTE - this is done only for probabilities from the last epoch. 
'''
def storeProbabilityFrequencyPlot(output_probabilities, plotsDirectoryPath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

    training_probabilities = output_probabilities["training"]

    training_probs_df = pd.DataFrame(data = training_probabilities, columns = ["output_probabilities"])
    sns.histplot(data = training_probs_df, x = "output_probabilities", ax = ax1)

    validation_probabilities = output_probabilities["validation"]
    vaidation_probs_df = pd.DataFrame(data = validation_probabilities, columns = ["output_probabilities"])
    sns.histplot(data = vaidation_probs_df, x = "output_probabilities", ax = ax2)

    plot_description = getParametersDescription()
    fig.text(.5, -0.1, plot_description, ha = 'center')
    plt.subplots_adjust(wspace = 0.3)
    ax1.set_xlim([-0.2, 1.2])
    ax2.set_xlim([-0.2, 1.2])
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_title("Predicted probabilities: Training")
    ax2.set_title("Predicted probabilities: Validation")
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "probabilityDistributionPlot")
        plt.savefig(plotPath, bbox_inches='tight')
    
    plt.show()
    plt.clf()

"""
Create a file(the name of the file has the current date and time) and save the state_dict of the model 
"""
def saveModel(model, plotsDirectoryPath):
    if(arguments["storePlots"]):
        filepath = os.path.join(plotsDirectoryPath, "modelState")
        f = open(filepath, "x")
        torch.save(model.state_dict(), filepath)
        f.close()

'''
Input - 
data - python dict (refer to the file description for more details on training or validation data) 
predicted_labels - A map with 2 keys - "training" and "validation". Each value is a list of labels (0 or 1), 
    where each val in the list is the label assigned to a single sample based on the probability (of being donor) 
    predicted by the model. Assigning labels are done using a threshold. 

Output - 
1. cf_matrix - A confusion matrix heat map of the number of samples that were true pos, true neg, false pos or false neg.
2. cf_matrix_labels - annotations in the cf matrix - like name, num_samples, percentage of samples in each category. 
'''
def getConfusionMatrixAndLabels(data, predicted_labels):
    labels = data["labels"]
    cf_matrix = confusion_matrix(labels, predicted_labels)

    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    cf_matrix_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    cf_matrix_labels = np.asarray(cf_matrix_labels).reshape(2,2)
    
    return cf_matrix, cf_matrix_labels

'''
Calls function to create confusion matrix, and uses this confusion matrix and labels to create a heatmap. 
'''
def storeConfusionMatrixHeatMap(trainingData, validationData, predicted_labels, plotsDirectoryPath):
    training_predictions = predicted_labels["training"]
    validation_predictions = predicted_labels["validation"]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 8))
    heatmap_description = getParametersDescription()
    fig.text(.5, -0.1, heatmap_description, ha = 'center', fontsize=12)
    
    training_cf_matrix, training_cf_matrix_labels = getConfusionMatrixAndLabels(trainingData, training_predictions)
    validation_cf_matrix, validation_cf_matrix_labels = getConfusionMatrixAndLabels(validationData, validation_predictions)
    s1 = sns.heatmap(training_cf_matrix, annot=training_cf_matrix_labels, fmt = '', cmap="Blues", ax=ax1, annot_kws={"fontsize":12})
    s2 = sns.heatmap(validation_cf_matrix, annot=validation_cf_matrix_labels, fmt = '', cmap="Blues", ax=ax2, annot_kws={"fontsize":12})
    s1.set_xlabel("Predicted Label", fontsize=12)
    s1.set_ylabel("True Label", fontsize=12)
    s2.set_xlabel("Predicted Label", fontsize=12)
    s2.set_ylabel("True Label", fontsize=12)
    fig.subplots_adjust(hspace=0.75, wspace=0.75)

    ax1.title.set_text(f'Training')
    ax2.title.set_text(f'Validation')

    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "confusionMatrix")
        plt.savefig(plotPath, bbox_inches='tight')
    
    plt.show()
    plt.clf()

'''
Splits a list of probabilities based on whether that sample was a true positive, true negative, false positive or false negative. 
Output - 
1. final_probs_df - A df with 2 columns - type and probabilities. Type can be true_pos, false_pos, true_neg or false_neg. 
2 description - Description that should go in the plot. 
'''
def getProbsForEachConfusionMatrixBlock(positive_probabilities, predicted_labels, true_labels):
    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []
    for i, true_label in enumerate(true_labels):
        if true_label == 1 and predicted_labels[i] == 1:
            true_pos.append(positive_probabilities[i])
        if true_label == 1 and predicted_labels[i] == 0:
            false_neg.append(positive_probabilities[i])
        if true_label == 0 and predicted_labels[i] == 1:
            false_pos.append(positive_probabilities[i])
        if true_label == 0 and predicted_labels[i] == 0:
            true_neg.append(positive_probabilities[i])

    # print(f"Printing confusion matrix individual block probabilities")
    # print(f"true positives probs: size is {len(true_pos)} and {true_pos[1:20]}")
    # print(f"false positives probs: size is {len(false_pos)} and {false_pos[1:20]}")
    # print(f"true negatives probs: size is {len(true_neg)} and {true_neg[1:20]}")
    # print(f"false negatives probs: size is {len(false_neg)} and {false_neg[1:20]}")

    true_pos_df = pd.DataFrame(true_pos, columns = ["probabilities"])
    true_pos_df["type"] = "true_pos"
    false_pos_df = pd.DataFrame(false_pos, columns = ["probabilities"])
    false_pos_df["type"] = "false_pos"
    true_neg_df = pd.DataFrame(true_neg, columns = ["probabilities"])
    true_neg_df["type"] = "true_neg"
    false_neg_df = pd.DataFrame(false_neg, columns = ["probabilities"])
    false_neg_df["type"] = "false_neg"

    df1 = pd.concat([true_pos_df, false_pos_df], ignore_index=True, axis=0)
    df2 = pd.concat([true_neg_df, false_neg_df], ignore_index=True, axis=0)
    final_probs_df = pd.concat([df1, df2], ignore_index=True, axis=0)

    description = f"True Pos: {len(true_pos)}, False Pos: {len(false_pos)}, True Neg: {len(true_neg)}, False Neg: {len(false_neg)} \n"
    return final_probs_df, description

'''
Generates the probability distribution plot, where x axis has probabilities ranging from 0 to 1 and y axis has the distribution. 
Here, probabilities refer to the softmax probabilities generated from the prediction of a model denoting the likelihood of 
a sample being donor-derived (label 1)
The probabiity distributions are plotted as separate curces as true positives, true negatives, false positives and false negatives. 
'''
def confsionMatrixLevelProbDistribtuionPlot(trainingData, validationData, output_probabilities, predicted_labels, plotsDirectoryPath):
    training_probs_df, training_desc = getProbsForEachConfusionMatrixBlock(output_probabilities["training"], 
                                                                        predicted_labels["training"], trainingData["labels"])
    validation_probs_df, validation_desc = getProbsForEachConfusionMatrixBlock(output_probabilities["validation"], 
                                                                        predicted_labels["validation"], validationData["labels"])

    fig_desc = "Training: "+ training_desc + "\n Validation: "+ validation_desc 

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 6))
    sns.kdeplot(data = training_probs_df, x = "probabilities", hue = "type", ax = ax1)
    sns.kdeplot(data = validation_probs_df, x = "probabilities", hue = "type", ax = ax2)

    fig.text(.5, -0.2, fig_desc, ha = 'center')
    ax1.title.set_text(f'Training Total epochs {arguments["numEpochs"]}')
    ax2.title.set_text(f'Validation Total epochs {arguments["numEpochs"]}')
    ax1.set_xlim([-0.2, 1.2])
    ax2.set_xlim([-0.2, 1.2])
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "confusionMatrixLevelProbabilityDistributionPlot")
        plt.savefig(plotPath, bbox_inches = "tight")
        
    plt.show()

'''
Plots learning rate over epochs. 
'''
def plotLearningRate(trainingData, plotsDirectoryPath):
    fig, ax = plt.subplots()
    learning_rates = trainingData["learningRates"]
    num_epochs = trainingData["epoch"] #This is the last epoch that was stored, so it will also be the number of epochs
    num_batches_single_epoch = len(learning_rates)/num_epochs  #Learning rate will get appended once per batch, for each epoch
    epoch_locations = [0]

    for i in range(1, num_epochs + 1):
        epoch_locations.append(epoch_locations[i-1] + num_batches_single_epoch)
    
    xs_learning_rate = [x for x in range(len(learning_rates))]
    plt.plot(xs_learning_rate, learning_rates)
    ax.vlines(x=epoch_locations, ymin = 0, ymax = max(learning_rates), colors='r', ls="--")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning rate curve used by the optimizer")

    # Add a legend to explain the red vertical lines outside the plot
    ax.axvline(0, color='r', ls="--", label='Start of an Epoch', linewidth=1)  # Example legend entry
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), frameon = True)

    fig.set_size_inches(12, 6)

    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "learningRatePlot")
        plt.savefig(plotPath, bbox_inches = "tight")

    plt.show()
    plt.clf()

'''
Store scipy generated performance report as a CSV file

Inputs
1. true_labels - List of true labels, each value represants true label (0/1) for a sample (for training or validation set).
2. predictions - list of probabilities output by the model, each value probability of sample being donor. 
3. sampleType - training or validation
4. plotsDirectoryPath - directory where the plots should be saved. 
'''
def storePerformanceMetrics(true_labels, predictions, sampleType, plotsDirectoryPath):
    target_names = ["donor", "recipient"]
    report = classification_report(true_labels, predictions, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(data = report).transpose()
    model_parameters_text = getParametersDescription()
    parameters_df = {'donor': 'model parameters', 'recipient': model_parameters_text}
    report_df = report_df.append(parameters_df, ignore_index = True)

    if(arguments["storePlots"]):
        filename = "performanceMetrics_" + sampleType + ".csv"
        csv_path =  os.path.join(plotsDirectoryPath, filename)
        report_df.to_csv(csv_path, index= True)

'''
Plot training and validation precision-recall curves. 
Inputs - 
1. training_positive_probs - list of predicted probs from the model, where each value represants the predicted prob for a 
    sample being donor-derived. 
2. training_true_labels - List of training true labels, each value represants true label (0/1) for a sample. 
3. validation_positive_probs - same as training_positive_probs but for validation set. 
4. validation_true_labels - same as training_true_labels but for validation set. 
5. plotsDirectoryPath - directory where the plots should be saved. 
'''
def plotPrecisionRecallCurve(training_positive_probs, training_true_labels, validation_positive_probs, validation_true_labels, plotsDirectoryPath):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))

    training_precision, training_recall, training_thresholds = precision_recall_curve(training_true_labels, training_positive_probs)
    ax1.plot(training_recall, training_precision)
    ax1.title.set_text('Training')

    validation_precision, validation_recall, validation_thresholds = precision_recall_curve(validation_true_labels, validation_positive_probs)
    ax2.plot(validation_recall, validation_precision)
    ax2.title.set_text('Validation')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    if(arguments["storePlots"]):
        plotPath = os.path.join(plotsDirectoryPath, "precisionRecallPlot")
        plt.savefig(plotPath, bbox_inches = "tight")
    
    plt.show()
    plt.clf()

#TODO To be removed once confusion matrix problems are fixed. 
def printDonorRecipientLabelsVsPredictions(true_labels, predictions, sampleType):
    true_count_0 = 0
    true_count_1 = 0

    for i in range(len(true_labels)):
        if(true_labels[i] == 0):
            true_count_0 = true_count_0 + 1
        if(true_labels[i] == 1):
            true_count_1 = true_count_1 + 1
    
    pred_count_0 = 0
    pred_count_1 = 0

    for i in range(len(predictions)):
        if(predictions[i] == 0):
            pred_count_0 = pred_count_0 + 1
        if(predictions[i] == 1):
            pred_count_1 = pred_count_1 + 1

    print(f"num of 0's and 1's predictions in the {sampleType} set is {pred_count_0} and {pred_count_1}")

def get_softmax(preds_array):
    preds_exp = np.exp(preds_array)
    sum_array = np.sum(preds_exp, axis = 1)
    softmax_preds = preds_exp/sum_array[:, None]
    return softmax_preds

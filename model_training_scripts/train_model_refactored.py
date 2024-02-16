'''
This file contains all functions for training a feed forward neural network model on Enformer predictions of cfDNA fragments
'''
import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import importlib
import torch
torch.cuda.empty_cache()

import numpy as np
import pandas as pd
import os
from datetime import datetime
import time

from torch.utils.data import DataLoader

import torch.optim as optim
from torch import nn
from torch.nn.functional import one_hot

import config
import plotUtils
import enformerOutputDataset
import enformerOutputDenseLayerModel as basicDenseLayer
import simpleDenseLayer

importlib.reload(basicDenseLayer)
importlib.reload(simpleDenseLayer)
importlib.reload(enformerOutputDataset)
importlib.reload(plotUtils)
importlib.reload(config)

arguments = {}
#Model hyperparameters
arguments["batchSize"] = config.modelHyperParameters.get("batchSize")
arguments["learningRate"] = config.modelHyperParameters.get("learningRate")
arguments["numberOfWorkers"] = config.modelHyperParameters.get("numberOfWorkers")
arguments["numberEpochs"] = config.modelHyperParameters.get("numberEpochs")
arguments["useCosineLearningFunction"] = config.modelHyperParameters.get("useCosineLearningFunction")

#Model general configs
arguments["trainingStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesTraining")
arguments["validationStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesValidation")
arguments["normalizeFeatures"] = config.modelGeneralConfigs.get("normalizeFeatures")
arguments["runWithControls"] = config.modelGeneralConfigs.get("runWithControls")

#Configs and file paths required for restoring from checkpoint
arguments["restoreFromCheckpoint"] = config.modelGeneralConfigs.get("restoreFromCheckpoint")
arguments["storePlots"] = config.modelGeneralConfigs.get("storePlots")
arguments["modelName"] = config.modelGeneralConfigs.get("modelName")
arguments["trainingAndValidationOutputsDirectory"] = config.filePaths.get("trainingAndValidationOutputsDirectory")
arguments["checkpointsFile"] = config.filePaths.get("checkpointsFile")
arguments["restoreCheckpointModelDirName"] = config.filePaths.get("restoreCheckpointModelDirName")

#General file paths
arguments["trainingEnformerTracksAverageFile"] = config.filePaths.get("trainingEnformerTracksAverageFile")
arguments["validationEnformerTracksAverageFile"] = config.filePaths.get("validationEnformerTracksAverageFile")

'''
The training loop for each epoch. It iteratates through the enformerOutput dataset which fetches enformer predictions,
divided into batches. If the function is called with isTraining true, then weights are updated in optimizer using
 backpropagation. If isTraining is false, predictions and loss are obtained for the data without updating the weights. 
 
Inputs - 
1. denseLayerModel - feed forward neural network object. 
2. dataloader - pytorch dataloader for the enformer output dataset. 
3. criterion - loss function
4. isTraining - true if its a training loop (weights are updated), false if validation
5. optimizer - optimizer object

Outputs - 
1. plotsData - dict with 3 keys - labels, inputData and predictions. Labels is list (0s and 1s) of labels (duh) for all 
    samples. predictions is a 2D list where col 1 is the predicted likelihood of a sample having label 1 (donor-derived)
    and col 2 is the prediced likelihood for label 0 (recipient-derived) 
2. avg_loss_per_batch - loss, but averaged over all the batches.
3. learningRates - list of learningRates for all the batches for this epoch (one per batch) (this is applicable only 
    if the optimizer is using cosine learning function, otherwise learning rate is constant for all epochs, all batches. 

'''
def trainModelGetPredictionsForEpoch(denseLayerModel, dataloader, criterion,
                              isTraining=False, optimizer = False):

    plotsData = {} #Dict with keys labels and predictions.
    learning_rates = []

    #TODO come up with a better way to initialize this rather than creating 0s arrays. This will also remove the slicing from the return statement
    modelPredictionsToRet = np.zeros(shape = (1, 2))
    modelInputLabelsToRet = []
    
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        enformerPrediction, classLabels = data
        if torch.cuda.is_available():
            #While creating torch.tensor, device can be passed as cuda. But that was a suspect for GPU node running out of memory.
            #After iterating through dataset and fetching each sample, send the labels and sequence to cuda
            #The class labels should be of type integer. 
            #TODO modify sampler function 
            #Because we use the sampler, there is an extra dimension for the labels and enformer output. [1*128*10626].
            #Take only the 1st element to remove the extra dimension. 
            enformerPrediction = enformerPrediction[0].to('cuda')
            classLabels = classLabels.to(torch.int64)[0].flatten().to('cuda')

        #The class labels have to be encoded into probabilities of type floating point
        probabilityLabels = one_hot(classLabels, num_classes=2).to(torch.float32)
        modelPrediction = denseLayerModel(enformerPrediction)
        loss = criterion(modelPrediction, probabilityLabels)
    
        #If the model is being trained, then do backpropagation and calculate loss. 
        if(isTraining):
            #zero grad is applicable only for optimizers and not for cosine annealing function schedulers. 
            if arguments["useCosineLearningFunction"] != True:
                optimizer.zero_grad()

            # Backward pass and calculate the gradients
            loss.backward()
            optimizer.step()

        #Collect data for plotting graphs
        running_loss += loss.item()
        if arguments["useCosineLearningFunction"] and isTraining:
            learning_rates.append(optimizer.get_lr())

        # modelInputDataToRet =  np.row_stack([modelInputDataToRet, enformerPrediction.detach().cpu().numpy()]) #For now we are not using the data anywhere, so adding this is not necessary. 
        modelInputLabelsToRet.extend(classLabels.cpu())
        modelPredictionsToRet = np.row_stack([modelPredictionsToRet, modelPrediction.detach().cpu().numpy()])
    
    avg_loss_per_batch = running_loss/len(dataloader)
    plotsData["labels"] = modelInputLabelsToRet
    plotsData["predictions"] = modelPredictionsToRet[1:, :]
    
    return plotsData, avg_loss_per_batch, learning_rates

'''
Master function that ties everything together (is the objectiveFn a misnomer?). 
1. The datasets, dataloaders, optimizers, loss functions etc are defined. 
2. Then the per epoch training function (above) is called to do backpropagation and optimize 
    weights per batch. 
3. After every epoch, the data generated by the model are saved along with the model checkpoints.
4. After the final training epoch, methods from plotUtils are called to plot performance related characteristics.

If restoreFromCheckpoints is true, weights from checkpoint directory (in the plots directory path) will be 
loaded onto the model before training. 

Inputs - 
learningRate, numWorkers, batchSize, numEpochs - hyperparameters that are defined in the config file. 
'''
def objectiveFn(learningRate, numWorkers, batchSize, numEpochs):
    '''
    Create the directories for plots and checkpoints. If restroring from checkpoint, plots directory is already created.
    Otherwise, create a new directory with a unique name.
    '''
    if(arguments["restoreFromCheckpoint"]):
        plots_directory_name = arguments["restoreCheckpointModelDirName"]
        plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"], plots_directory_name)
    else:
        now = datetime.now()
        filename_extension = now.strftime("%d_%m_%H_%M_%S")
        plots_directory_name = filename_extension + "_" + str(arguments["modelName"])
        plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"], plots_directory_name)
        os.mkdir(plots_directory_path)

    denseLayerModel = basicDenseLayer.BasicDenseLayer().to('cuda')

    '''
    The model can be trained on simulated data with augmented signals. The details of creating the simulated data 
    are present in the EnformerOutputDataset class. For creating simulations, the minimum and maximum value for each
    enformer track are required. These values are pre-caculated using a separate script and stored in CSV files. Load this
    data and pass to the enformerOutputDataset for creating simulations. 
    '''
    if(arguments["runWithControls"] == True):
        training_enformer_track_ranges = pd.read_csv(arguments["trainingEnformerTracksAverageFile"], sep='\t')
        validation_enformer_track_ranges = pd.read_csv(arguments["validationEnformerTracksAverageFile"], sep='\t')
        
        percent_features =  arguments["percentageFeaturesAsControls"]
        percent_samples = arguments["percentageSamplesAsControls"]
        print(f"Running the model with controls, about to create datasets. percent_aug_feats: {percent_features} and percent_aug_samples: {percent_samples}")
        trainingDataset = enformerOutputDataset.EnformerOutputDataset("training", arguments["normalizeFeatures"], training_enformer_track_ranges, 
                                                                      percent_features, percent_samples)
        
        validationDataset = enformerOutputDataset.EnformerOutputDataset("validation", arguments["normalizeFeatures"], validation_enformer_track_ranges, 
                                                                        percent_features, percent_samples)

    else:
        print(f"The model is not running with simulated data. Reading Enformer output files to get the data for training. ")
        trainingDataset = enformerOutputDataset.EnformerOutputDataset("training", arguments["normalizeFeatures"])
        validationDataset = enformerOutputDataset.EnformerOutputDataset("validation", arguments["normalizeFeatures"])


    print(f"Num of training samples: {len(trainingDataset)} and validation samples: {len(validationDataset)}")

    #Get training dataloader
    rangeTrainingSampler = range(arguments["trainingStartIndex"] , len(trainingDataset) + arguments["trainingStartIndex"])
    trainingsampler = torch.utils.data.BatchSampler(rangeTrainingSampler, batch_size=batchSize,
                                            drop_last=False )
    trainingDataloader = DataLoader(trainingDataset,  num_workers=numWorkers, sampler=trainingsampler)

    #Get validation dataloader
    rangeValidationSampler = range(arguments["validationStartIndex"] , len(validationDataset) + arguments["validationStartIndex"])
    validation_sampler = torch.utils.data.BatchSampler(rangeValidationSampler, batch_size=batchSize,
                                            drop_last=False )
    validationDataloader = DataLoader(validationDataset, num_workers=numWorkers, sampler=validation_sampler)

    #Get loss function
    training_class_weights = trainingDataset.getClassWeights()
    criterion = nn.CrossEntropyLoss(weight = torch.tensor(training_class_weights).to('cuda'))
    # criterion = nn.BCEWithLogitsLoss()

    #If restoring from a previous checkpoint, load the model and optimizer state dict 
    epoch_to_start = 1

    #If restoring from checkpoint, load the state dict onto the dense layer model. If this is the case, also set the start_epoch.
    if(arguments["restoreFromCheckpoint"]):
        checkpoint_path = os.path.join(plots_directory_path, arguments["checkpointsFile"])
        checkpoint_dict = torch.load(checkpoint_path)
        denseLayerModel.load_state_dict(checkpoint_dict["model_state_dict"])
        epoch_to_start = checkpoint_dict["epoch"] + 1
        print(f"Restore from checkpoint is True.. loading previous model checkpoint and starting from epoch : {epoch_to_start}")
    
    training_num_batches = len(trainingDataloader)
    optimizer = optim.Adam(denseLayerModel.parameters(), lr=learningRate)
    # optimizer = optim.SGD(denseLayerModel.parameters(), lr=learningRate)

    if(arguments["restoreFromCheckpoint"]):
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    if arguments["useCosineLearningFunction"]:
        optimizer_steps = (training_num_batches * numEpochs) #Number of steps in gradient descent. 
        optimizer_to_use_for_training = optim.lr_scheduler.CosineAnnealingLR(optimizer, optimizer_steps, last_epoch = epoch_to_start - 2, eta_min=0)
    else:
        optimizer_to_use_for_training = optimizer

    #Train model and validate it for each epoch
    for epoch in range(epoch_to_start, numEpochs + 1):
        print(f"Starting training for epoch {epoch}")
        trainingPlotsData, training_loss, training_learning_rates = trainModelGetPredictionsForEpoch(denseLayerModel, 
                                            trainingDataloader, criterion, isTraining=True, optimizer=optimizer_to_use_for_training)
        trainingPlotsData["epoch"] = epoch
        print(f"Finished training for epoch {epoch}. Starting validations")
        
        #Validation
        with torch.no_grad():
            validationPlotsData, validation_loss, validation_learning_rates = trainModelGetPredictionsForEpoch(denseLayerModel, 
                                                validationDataloader, criterion, isTraining=False)
            validationPlotsData["epoch"] = epoch
            
        #For every epoch, save the model checkpoint and the plotsData so far. 
        plotUtils.saveModelCheckpoint(epoch, denseLayerModel, optimizer, plots_directory_path)
        plotUtils.savePlotsData("training", trainingPlotsData, training_loss, training_learning_rates, plots_directory_path)
        plotUtils.savePlotsData("validation", validationPlotsData, validation_loss, validation_learning_rates, plots_directory_path)

    print(f"Completed training and validation. Saving model and plotting loss function graphs. ")
    plotUtils.storePlots(plots_directory_path, modelInputType="Enformer")

if __name__ == '__main__':
    print(f"Start time is {time.time()}")
    learningRate = arguments["learningRate"]
    numWorkers = arguments["numberOfWorkers"]
    numEpochs = arguments["numberEpochs"]
    batchSize = arguments["batchSize"]

    print(f"The model hyper parameters are Learning Rate: {learningRate}, numWorkers: {numWorkers}, numEpochs: {numEpochs}, batchSize: {batchSize}")

    objectiveFn(learningRate, numWorkers, batchSize, numEpochs)

    '''
    For simulations, the following code can be used to train multiple models iteratively for varying percentages of
    features and samples with augmented signals. 
    '''
    print(f"The model hyper parameters are Learning Rate: {learningRate}, numWorkers: {numWorkers}, numEpochs: {numEpochs}, batchSize: {batchSize}")

    # Iterate over each possible % samples and % features augmentation, train the model and store plots.
    existing_percents = [0, 2, 5, 10, 30, 50, 70, 90]
    extra_sample_to_cover = [0, 2]
    extra_feature_to_cover = [0, 2]

    # Run the model for the extra samples to cover
    for sample_percent in extra_sample_to_cover:
        for feature_percent in existing_percents:
            print(f"Starting the model for Sample percent : {sample_percent} and feature percent : {feature_percent}")
            model_name = f"sample_{sample_percent}_feature_{feature_percent}_percent"
            arguments["percentageFeaturesAsControls"] = feature_percent
            arguments["percentageSamplesAsControls"] = sample_percent

            # Create the directories for plots and checkpoints
            if (arguments["restoreFromCheckpoint"]):
                plots_directory_name = arguments["restoreCheckpointModelDirName"]
                plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"],
                                                    plots_directory_name)
            else:
                now = datetime.now()
                filename_extension = now.strftime("%d_%m_%H_%M_%S")
                plots_directory_name = filename_extension + "_" + model_name
                plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"],
                                                    plots_directory_name)
                os.mkdir(plots_directory_path)

            objectiveFn(learningRate, numWorkers, batchSize, numEpochs, plots_directory_path)

    print(f"Finished making predictions for all features for the extra sample percent")
    print(f"Current time is {time.time()}")

    for feature_percent in extra_feature_to_cover:
        for sample_percent in existing_percents:
            print(f"Starting the model for Sample percent : {sample_percent} and feature percent : {feature_percent}")
            model_name = f"sample_{sample_percent}_feature_{feature_percent}_percent"
            arguments["percentageFeaturesAsControls"] = feature_percent
            arguments["percentageSamplesAsControls"] = sample_percent

            # Create the directories for plots and checkpoints
            if (arguments["restoreFromCheckpoint"]):
                plots_directory_name = arguments["restoreCheckpointModelDirName"]
                plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"],
                                                    plots_directory_name)
            else:
                now = datetime.now()
                filename_extension = now.strftime("%d_%m_%H_%M_%S")
                plots_directory_name = filename_extension + "_" + model_name
                plots_directory_path = os.path.join(arguments["trainingAndValidationOutputsDirectory"],
                                                    plots_directory_name)
                os.mkdir(plots_directory_path)

            objectiveFn(learningRate, numWorkers, batchSize, numEpochs, plots_directory_path)
    print(f"End time is {time.time()}")
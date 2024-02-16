'''
This is a test Jupyter file with functions to train Deep Learning model 3: The CNN model that extracts sequence motifs
from cfDNA fragment sequences, and uses these sequence motifs in combination with Enformer predictions to classify the
fragment as donor-derived or recipient-derived.
'''
import sys
sys.path.insert(0,'/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts')
import importlib
import torch
torch.cuda.empty_cache()

import numpy as np
import os
from datetime import datetime
import time

from torch.utils.data import DataLoader

import torch.optim as optim
from torch import nn
from torch.nn.functional import one_hot

import config
import plotUtils
import combinedSequenceEnfomerDataset
import combinedSequenceEnformerModel

importlib.reload(plotUtils)
importlib.reload(config)
importlib.reload(combinedSequenceEnformerModel)
importlib.reload(combinedSequenceEnfomerDataset)

arguments = {}

#Model hyperparameters
arguments["batchSize"] = config.modelHyperParameters.get("batchSize")
arguments["learningRate"] = config.modelHyperParameters.get("learningRate")
arguments["numberOfWorkers"] = config.modelHyperParameters.get("numberOfWorkers")
arguments["numberEpochs"] = config.modelHyperParameters.get("numberEpochs")
arguments["useCosineLearningFunction"] = config.modelHyperParameters.get("useCosineLearningFunction")
arguments["inputSizeToCnn"] = config.modelHyperParameters.get("sequenceDatasetOutputSize")
arguments["dropoutProbability"] = config.modelHyperParameters.get("dropoutProbability")
arguments["weightDecay"] = config.modelHyperParameters.get("weightDecay")

#Enformer related configs
arguments["trainingStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesTraining")
arguments["validationStartIndex"] = config.modelGeneralConfigs.get("startIndexEnformerSamplesValidation")

#Model general configs
arguments["addLengthAsFeature"] = config.modelGeneralConfigs.get("addLengthAsFeature")

#Configs and file paths required for restoring from checkpoint
arguments["restoreFromCheckpoint"] = config.testConfigs.get("restoreFromCheckpoint")
arguments["storePlots"] = config.testConfigs.get("storePlots")
arguments["modelName"] = config.modelGeneralConfigs.get("modelName")
arguments["trainingAndValidationOutputsDirectory"] = config.filePaths.get("trainingAndValidationOutputsDirectory")
arguments["checkpointsFile"] = config.filePaths.get("checkpointsFile")
arguments["restoreCheckpointModelDirName"] = config.filePaths.get("restoreCheckpointModelDirName")

'''
The training loop for each epoch. It iteratates through the both sequenceDataset (returns cfDNA fragment sequences) and 
enformerOutputDataset (with returns 
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
def trainCombinedModelAndGetPredictions(combinedSequenceEnformerModel, dataloader, criterion, 
                              isTraining=False, optimizer = False):
    plotsData = {} 
    
    modelPredictionsToRet = np.zeros(shape = (1, 2))
    modelInputLabelsToRet = []
    learningRates = []
    
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        sequenceOutput, enformerOutput, og_sequence_length, classLabels = data
        if torch.cuda.is_available():
            #While creating torch.tensor, device can be passed as cuda. But that was a suspect for GPU node running out of memory.
            #After iterating through dataset and fetching each sample, send the labels and sequence to cuda
            #The class labels should be of type integer. 
            #TODO modify sampler function 
            #Because we use the sampler, there is an extra dimension for the labels and enformer output. [1*128*10626].
            #Take only the 1st element to remove the extra dimension. 
            enformerOutput = enformerOutput[0].to('cuda')
            sequenceOutput = sequenceOutput[0].to('cuda')
            batches, sequence_length, one_hot_base_length = sequenceOutput.shape
            sequenceOutput = sequenceOutput.reshape(batches, 1, sequence_length, one_hot_base_length)
            classLabels = classLabels.to(torch.int64)[0].flatten().to('cuda')
            og_sequence_length =  og_sequence_length[0].reshape(len(og_sequence_length[0]), 1).to("cuda")

        #The class labels have to be encoded into probabilities of type floating point
        probabilityLabels = one_hot(classLabels, num_classes=2).to(torch.float32)

        modelPrediction = combinedSequenceEnformerModel(sequenceOutput, enformerOutput, og_sequence_length, arguments["addLengthAsFeature"])
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
                learningRates.append(optimizer.get_lr())
        # modelInputDataToRet =  np.row_stack([modelInputDataToRet, enformerOutput.detach().cpu().numpy()])
        modelInputLabelsToRet.extend(classLabels.cpu())
        modelPredictionsToRet = np.row_stack([modelPredictionsToRet, modelPrediction.detach().cpu().numpy()])
    
    avg_loss_per_batch = running_loss/len(dataloader)
    plotsData["labels"] = modelInputLabelsToRet
    plotsData["predictions"] = modelPredictionsToRet[1:, :]

    return plotsData, avg_loss_per_batch, learningRates

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

    denseLayerModel = combinedSequenceEnformerModel.combinedSequenceEnformerModel(arguments["dropoutProbability"]).to('cuda')
    
    #Training dataloader
    trainingDataset = combinedSequenceEnfomerDataset.combinedSequenceEnfomerDataset("training")
    rangeTrainingSampler = range(arguments["trainingStartIndex"] , len(trainingDataset) + arguments["trainingStartIndex"])
    trainingsampler = torch.utils.data.BatchSampler(rangeTrainingSampler, batch_size=batchSize,
                                            drop_last=False )
    trainingDataloader = DataLoader(trainingDataset,  num_workers=numWorkers, sampler=trainingsampler)

    #Validation dataloader TODO uncomment once validation enformer output file is ready. 
    validationDataset = combinedSequenceEnfomerDataset.combinedSequenceEnfomerDataset("validation")
    rangeValidationSampler = range(arguments["validationStartIndex"] , len(validationDataset) + arguments["validationStartIndex"])
    validation_sampler = torch.utils.data.BatchSampler(rangeValidationSampler, batch_size=batchSize,
                                            drop_last=False)
    validationDataloader = DataLoader(validationDataset, num_workers=numWorkers, sampler=validation_sampler)

    print(f"Number of training samples and validation samples are {len(trainingDataset)} and {len(validationDataset)}")
    #Get loss function
    training_class_weights = trainingDataset.getClassWeights()
    criterion = nn.CrossEntropyLoss(weight = torch.tensor(training_class_weights).to('cuda'))

    #If restoring from a previous checkpoint, load the modek and optimizer state dict 
    epoch_to_start = 1

    if(arguments["restoreFromCheckpoint"]):
        checkpoint_path = os.path.join(plots_directory_path, arguments["checkpointsFile"])
        checkpoint_dict = torch.load(checkpoint_path)
        denseLayerModel.load_state_dict(checkpoint_dict["model_state_dict"])
        epoch_to_start = checkpoint_dict["epoch"] + 1
        print(f"Restore from checkpoint is True.. loading previous model checkpoint and starting from epoch : {epoch_to_start}")
    
    training_num_batches = len(trainingDataloader)
    optimizer = optim.Adam(denseLayerModel.parameters(), lr=learningRate)

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
        trainingPlotsData, training_loss, training_learning_rates = trainCombinedModelAndGetPredictions(denseLayerModel, 
                                            trainingDataloader, criterion, isTraining=True, optimizer=optimizer_to_use_for_training)
        trainingPlotsData["epoch"] = epoch
        print(f"Finished training for epoch {epoch}. Starting validations")
        
        #Validation
        with torch.no_grad():
            validationPlotsData, validation_loss, validation_learning_rates = trainCombinedModelAndGetPredictions(denseLayerModel, 
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
    print(f"End time is {time.time()}")

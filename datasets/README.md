### Description and purpose of all the scripts in this folder 

  This folder contains the three pytorch datasets that are the inputs for training the three Deep Learning models. Pytorch input datasets are iterable objects. Upon iterating, they call the __get_item() function within the dataset, which are written to read from different sources, process the input and return data in the format needed for training the models. They return n samples, where n is the batch-size configured in the dataset. 


| S.No | Script name | Description of contents | Purpose |
|------|-------------|-------------------------|---------|
| 1 | [combinedSequenceEnfomerDataset.py](https://github.com/vmadhupreetha/fragmentomics/blob/master/datasets/combinedSequenceEnfomerDataset.py) | Pytorch dataset that returns Enformer-predicted tracks and one-hot encoded sequence, which are in turn read from Enformer output H5PY files and one-hot encoded sequence H5PY files. | The input dataset for training and validating Deep Learning model 3 (Combined CNN model) |
| 2 | [enformerOutputDataset.py](https://github.com/vmadhupreetha/fragmentomics/blob/master/datasets/enformerOutputDataset.py) | Pytorch dataset that returns Enformer-predicted tracks, which are in turn read from Enformer output H5PY files. | The input dataset for training and validating Deep Learning model 1 (simple feed forward neural network model) |
| 3 | [sequenceDataset.py](https://github.com/vmadhupreetha/fragmentomics/blob/master/datasets/sequenceDataset.py) | Pytorch dataset that returns one-hot encoded sequences, which are generated from processing coordinate H5PY files. | The input dataset for training and validating Deep Learning Model 3 (CNN model) |


import numpy as np
import h5py

def shuffle_enformer_output():
    sampleType = "validation"
    enformer_file = f"/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_final_model_125k_non_shuffled.hdf5"
    new_h5_file_path = f"/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/validation_final_model_125k.hdf5"

    new_h5_file = h5py.File(new_h5_file_path, "w-")

    with h5py.File(enformer_file, 'r') as f:
        numSamples = len(f[f"{sampleType}Labels"][:])

    enformerOutputDatasetName = f"{sampleType}EnformerOutput"
    labelsDatasetName = f"{sampleType}Labels"

    numEnformerOuputSingleSample = 2*5313

    new_h5_file.create_dataset(enformerOutputDatasetName,  (numSamples, numEnformerOuputSingleSample),
                                        compression="gzip", compression_opts=8, chunks = (200, numEnformerOuputSingleSample))
    new_h5_file.create_dataset(labelsDatasetName,  (numSamples, 1),
                                        compression="gzip", compression_opts=8, chunks = (200, 1))

    with h5py.File(enformer_file, 'r') as f:
        labels = f[labelsDatasetName][:]
        enformerOutput = f[enformerOutputDatasetName][:]
        np.random.seed(42)
        np.random.shuffle(enformerOutput)
        np.random.seed(42)
        np.random.shuffle(labels)

    new_h5_file[enformerOutputDatasetName][0:numSamples, :] = enformerOutput
    new_h5_file[labelsDatasetName][0:numSamples, :] = labels

    new_h5_file.close()
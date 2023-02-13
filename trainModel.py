import get_data
import pass_to_enformer
from torch.utils.data import Dataset, DataLoader

#Data will be a list of tuples. Each tuple in the list will be the data for a single cfDNA fragment. The values in the
#tuple are [chromosome_number, start, end and the sequence of the cfDNA fragment)
data = get_data.getCfdnaSequence()

#Get only the 1st training sample
first_sample_sequence = data[0]
print(f"First sample is {first_sample_sequence}")
pass_to_enformer.getEnformerPredictions(first_sample_sequence)

# for index, single_fragment_data in enumerate(data):
#     chrNum = single_fragment_data[0]
#     start = single_fragment_data[1]
#     end = single_fragment_data[2]
#     sequence = single_fragment_data[3]
#
#     enformerOutput = pass_to_enformer.getEnformerPredictions(sequence)


# def getLabelForSequence:
#
#     """
#     Get the list of SNVs for the patient involved. How do we connect the cfDNA fragments with a patient ?
#     #How to align the cfDNA fragment with the reference genome ? We have to pass the coordinates from the bed file.
#
#     """
# class EnformerDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         self.data.append()




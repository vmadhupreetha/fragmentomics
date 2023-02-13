import numpy as np

import torch
from enformer_pytorch import Enformer, seq_indices_to_one_hot
import get_data

# #Is this pretrained model ?
# model = Enformer.from_hparams(
#     dim = 1536,
#     depth = 11,
#     heads = 8,
#     output_heads = dict(human = 5313, mouse = 1643),
#     target_length = 896,
# )
#
# sequence = torch.randint(0, 5, (1, 196_608)) #pass almost 200 kilo base pairs to Enformer
# output = model(sequence)
#
# human_output_tensor = output['human']
#
# #torch.Size([1, 896, 5313])
# #896 values giving the values of epigenomic features. 5313 tracks.
# print(human_output_tensor.size())

#Sequence can also be passed as one hot encodings.
# seq = torch.randint(0, 5, (1, 200))
# print(f"The sequence is {seq}")
# one_hot = seq_indices_to_one_hot(seq)
# print(f"One hot encoding type is {type(one_hot)} and actual value is {one_hot}")

def getEnformerPredictions(data):

    #If we dont convert to torch to int64, then torch.nn.functional.one_hot throws this error
    #RuntimeError: one_hot is only applicable to index tensor.
    # first_sample_tensor = torch.tensor(data[0]).to(torch.int64)

    print(f"Printing the size of the np array {data.size}")
    first_sample_tensor = torch.tensor(np.float32(data))
    print(first_sample_tensor)


    # """
    # If we use the custom seq to one hot encoding function from get_data.py, then Enformer is throwing this error
    # RuntimeError: expected scalar type Double but found Float
    # Acc to torch messages, double is basically float 64. Tried converting the tensor to double using .double(). Same error
    # """
    # one_hot_sequence = seq_indices_to_one_hot(first_sample_tensor)
    # print("Finished doing the one hot encoding")
    # print(type(one_hot_sequence))

    print("Passing the sequence through pre trained enformer")
    enformer_pretrained = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_checkpointing = True)
    pretrained_output = enformer_pretrained(first_sample_tensor)
    human_pretrained_output = pretrained_output['human']
    print(f"Finished predicting from enformer, the size is {human_pretrained_output.size()}")
    print(f"Printing the size of the 1st track {human_pretrained_output[:, 0].size()}")
    return human_pretrained_output
import numpy as np
import pandas as pd

import h5py

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import sklearn.metrics as metric

import math
import random

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
# import umap
# import umap.plot

#Read Enformer output files and get the track for plotting
enformerOutputFile = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/output/EnformerOutputs/enformer_output_validation_class_balanced.hdf5"
with h5py.File(enformerOutputFile, 'r') as f:
    samples = f["validationEnformerOutput"][:]
    print(samples.shape)

print(f"Finished loading all the samples")

#Split Enformer tracks into donors and recipients
with h5py.File(enformerOutputFile, 'r') as f:
    labels = f["validationLabels"][:]
    recip_indices = np.where(labels == 1)[0]
    donor_indices = np.where(labels == 0)[0]

donors = samples[donor_indices]
recips = samples[recip_indices]

num_donors = len(donors)
num_recips = len(recips)
print(f"Num donors is {num_donors} and number of recipients is {num_recips}")

#Combine the donors and recipients such that all donors are grouped together. 
combined_samples = np.vstack((donors, recips))
num_tracks = combined_samples.shape[1]

columnNames = []
for i in range(1, num_tracks+1):
    columnNames.append("EnformerTrack" + str(i))

#Make into a dataframe, scale and add class labels as a column
donor_recip_enformer_tracks_df = pd.DataFrame(combined_samples, columns = columnNames)
donor_recip_enformer_tracks_df.loc[0:num_donors-1, "target"] = 1
donor_recip_enformer_tracks_df.loc[0:num_donors-1, "target_name"] = "donor"
donor_recip_enformer_tracks_df.loc[num_donors:num_donors + num_recips - 1, "target"] = 0
donor_recip_enformer_tracks_df.loc[num_donors:num_donors + num_recips -1, "target_name"] = "recipient"

print(f"Printing the combined df after adding classes")
print(donor_recip_enformer_tracks_df.head(10))

#Scale the tracks 
donor_recip_enformer_tracks_df_scaled = StandardScaler().fit_transform(donor_recip_enformer_tracks_df)
print(f"Finished getting the scaled df")

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(donor_recip_enformer_tracks_df_scaled)
print("Finished getting TSNE results")

print(tsne_results.head(10))

tsne_results.loc[0:num_donors-1, "target"] = 1
tsne_results.loc[0:num_donors-1, "target_name"] = "donor"
tsne_results.loc[num_donors:num_donors + num_recips - 1, "target"] = 0
tsne_results.loc[num_donors:num_donors + num_recips -1, "target_name"] = "recipient"

# Create a scatter plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Define colors for "donor" and "recipient"
colors = {'donor': 'blue', 'recipient': 'red'}

# Loop through the data points and plot them with colors based on "target_name"
for target_name, color in colors.items():
    subset = tsne_results[tsne_results['target_name'] == target_name]
    plt.scatter(subset[0], subset[1], label=target_name, c=color, alpha=0.5)

# Add labels and legend
plt.title("t-SNE Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()

# Show the plot
initial_analysis_plots_dir = "/hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts/one_time_use_side_scripts/enformer_tracks_initial_analysis_plots"
plotPath = os.path.join(initial_analysis_plots_dir, "tsne_plot")
plt.savefig(plotPath, bbox_inches='tight')

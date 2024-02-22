# Fragmentomics

Hey there !! Welcome to the GitHub repository for my project titled "Developing Deep Learning models for predicting lung transplant rejections using cell-free DNA fragmentomics approaches". 

This project was conducted as part of my Masters studies in Bioinformatics and Bicomplexity at Utrecht University. I developed three deep learning models to learn patterns distinguishing donor-derived cell-free (dd-cfDNA) DNA from recipient-derived cell-free DNA (rd-cfDNA). This repository contains the scripts developed during my research, for data pre-processing, data analysis, training the models and evaluating their performance. Feel free to explore the repository and reach out to me at madhuvl96@gmail.com, if you have any questions or would like to collaborate :) 

## Motivation

  Chronic respiratory diseases(CRD) impose a significant burden to society responsible for around 4 million deaths globally every year. Lung transplant remains the last resort for patients with CRD, when all other treatment options have failed. However, lung transplant recipients suffer from poor long term survival rates, with average 5-year survival rates as low as 50%. Traditional methods for diagnosing rejection, like transbronchial biopsy are invasive and often prove too late. Thus, there is an urgent need for developing diagnostic methods for timely detection of lung transplant rejections, in order to improve lung transplant survival outcomes. 

  cell-free DNA(cfDNA) are small fragments of DNA found in human blood, that were released by dying cells as a result of processes like apoptosis. Most of these fragments arise from blood cells. However, in special conditions like tumour and tissue injury, the affected tissue also releases cfDNA into the blood. The transplanted lung is usually subjected to immune response from the recipient's system, which causes it to release cell-free DNA into the blood, called donor-derived cell-free DNA (dd-cfDNA). In the event of chronic rejection, more dd-cfDNA is released owing to more severe tissue injury. Thus the levels of dd-cfDNA in the recipient's blood is indicative of the transplant status and holds great promose as a biomarker for diagnosing rejections. 

  The main challenge with measuring the levels of dd-cfDNA is differentiating between dd-cfDNA and the recipient's own cell-free DNA (recipient-derived cfDNA or rd-cfDNA). Current methods to measure dd-cfDNA leverage Single Nucleotide Polymorphism (SNP) differences between the donor and the recipient. Markers specific to donor SNPs are used to selectively amplify dd-cfDNA. However, this method is limited by the availability of donor information and the number of SNP differences between the donor and the recipient. 

  To mitigate this limitation, we've proposed a SNP-free approach of differentiating between dd-cfDNA and rd-cfDNA based on their tissue of origin. Since the majority of cell-free DNA in a healthy individual originate from blood cells, rd-cfDNA predominantly originates from blood cells whereas dd-cfDNA originats from lung cells. We've trained Deep Learning models to learn patterns indicative of the tissue of origin, to differentiate between dd-cfDNA and rd-cfDNA. 

## Final Aim 

  Develop and train deep learning models to distinguish between donor- and recipient-derived cfDNA, and inturn use these models to predict whether lung transplant recipients are experiencing rejection based on their cfDNA sequence. 

## Input dataset 
  The dataset for this project was sourced from a study conducted on 47 lung transplant recipients by De Vlaminck et al. in 2015 [[1]](https://www.pnas.org/doi/abs/10.1073/pnas.1517494112). The data consists of cfDNA fragment sequences collected from these patients at various time points post-transplant. The labels for each cfDNA fragment comes from the SNP-genotyping based classification performed by the authors in the study. 

<img width="1209" alt="the_dataset" src="https://github.com/vmadhupreetha/fragmentomics/assets/113985193/3653cc34-c8d7-4ff7-9764-7a2971d50aec">

## Methodology :

### Features for training

Given than all cell-types share the same sequence, cell-free DNA sequence alone is not sufficient to train models to differentiate between cell types. Fragmentation of DNA during cfDNA formation is not random, but are influenced by the epigenetic landscape of the parent cells. Consequently, different regions of the genome form cell-free DNA in different cell types [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4715266/). We explored two feature sets that leverage the fact that donor- and recipient-derived fragments originate from different genomic regions.

- **Enformer-generated epigenetic tracks as feature sets:**<br>
    The general idea behind this feature set is that different genomic regions inturn differ in their chromatin structure, accessibility, epigenetic modifications. For capturing these properties as numerical values for training the models, we used a pre-trained Deep Learning model called ["Enformer"](https://github.com/lucidrains/enformer-pytorch). Enformer takes a DNA sequence as input and predicts it's epigenetic properties in the form of the simulated results of 4 experimenets - CAGE-seq, ATAC-seq, CHIP-seq and DNAse accessibility. The predictions are presented in the form of 5,313 epigenetic "tracks", where each track refers to the simulated results of one of these four experiment types for one cell type [[3]](https://www.nature.com/articles/s41592-021-01252-x). 

- **Sequence motifs**<br>
   Motifs are recurring patterns of nucleotides that are usually associated with functional elements like promoters, enhancers, or coding regions. Owing to differences in functionality, it is highly likely that there are differences in the sequence motifs extracted from different genomic regions.

### The Deep Learning models: 
  We trained three Deep Learning models to use one or a combination of the two feature sets described above. 


| S.No | Model type                              | Input data                                      | Description                                                                                                   |
|------|-----------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 1.   | Feedforward neural network              | 5,313 Enformer-generated epigenetic tracks     | Classifies a given cfDNA fragment as donor- or recipient-derived using Enformer-generated epigenetic tracks as features |
| 2.   | Convolutional neural network (CNN)     | cfDNA sequence                                  | Extracts motifs from the cfDNA sequence and classifies the fragment as donor- or recipient-derived based on these motifs. |
| 3.   | Convolutional Neural Network (CNN)     | cfDNA sequence and 5,313 Enformer-generated epigenetic tracks | Extracts motifs from sequences, uses both sequence motifs and Enformer-generated epigenetic tracks for classifying fragments as donor- or recipient-derived. |


<img width="1095" alt="flowchart_deep_leaning_models" src="https://github.com/vmadhupreetha/fragmentomics/assets/113985193/544a846f-dc14-4001-a901-61a0e8519936">

### Training and validation workflow 

  The dataset were split into training, validation and test sets. The three Deep Learning models described above were trained with the training set. Validation set was used for hyperparameter tuning and to compare the performance of the three models against each other. The best performing model was chosen as the one with the highest AUC score on the validation set. The best performing model was used to get the predictions of the test patient set. These predictions were compared against the results from SNP genotyping. 

<img width="1134" alt="workflow" src="https://github.com/vmadhupreetha/fragmentomics/assets/113985193/e9aa587e-9197-46d2-9fbf-bf4aea2f542c">

## Additional Information

Refer to the [thesis report](https://github.com/vmadhupreetha/fragmentomics/blob/master/fragmentomics_thesis_report.pdf) or [poowerpoint slides](https://github.com/vmadhupreetha/fragmentomics/blob/master/fragmentomics_presentation.pptx) for more information.

## References 
1. De Vlaminck, Iwijn, et al. "Noninvasive monitoring of infection and rejection after lung transplantation." Proceedings of the National Academy of Sciences 112.43 (2015): 13336-13341 <br>
2. Snyder, Matthew W., et al. "Cell-free DNA comprises an in vivo nucleosome footprint that informs its tissues-of-origin." Cell 164.1 (2016): 57-68 <br>
3. Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203 <br>





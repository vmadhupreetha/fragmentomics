# Fragmentomics

Hey there !! Welcome to the GitHub repository for my project titled "Developing Deep Learning models for predicting lung transplant rejections using cell-free DNA fragmentomics approaches". 

This project was conducted as part of my Masters studies in Bioinformatics and Bicomplexity at Utrecht University. I developed three deep learning models to learn patterns distinguishing donor-derived cell-free (dd-cfDNA) DNA from recipient-derived cell-free DNA (rd-cfDNA). This repository contains the scripts developed during my research, for data pre-processing, data analysis, training the models and evaluating their performance. Feel free to explore the repository and reach out to me at madhuvl96@gmail.com, if you have any questions or would like to collaborate :) 

### Motivation

  Chronic respiratory diseases(CRD) impose a significant burden to society responsible for around 4 million deaths globally every year. Lung transplant remains the last resort for patients with CRD, when all other treatment options have failed. However, lung transplant recipients suffer from poor long term survival rates, with average 5-year survival rates as low as 50%. Traditional methods for diagnosing rejection, like transbronchial biopsy are invasive and often prove too late. Thus, there is an urgent need for developing diagnostic methods for timely detection of lung transplant rejections, in order to improve lung transplant survival outcomes. 

  cell-free DNA(cfDNA) are small fragments of DNA found in human blood, that were released by dying cells as a result of processes like apoptosis. Most of these fragments arise from blood cells. However, in special conditions like tumour and tissue injury, the affected tissue also releases cfDNA into the blood. The transplanted lung is usually subjected to immune response from the recipient's system, which causes it to release cell-free DNA into the blood, called donor-derived cell-free DNA (dd-cfDNA). In the event of chronic rejection, more dd-cfDNA is released owing to more severe tissue injury. Thus the levels of dd-cfDNA in the recipient's blood is indicative of the transplant status and holds great promose as a biomarker for diagnosing rejections. 

  The main challenge with measuring the levels of dd-cfDNA is differentiating between dd-cfDNA and the recipient's own cell-free DNA (recipient-derived cfDNA or rd-cfDNA). Current methods to measure dd-cfDNA leverage Single Nucleotide Polymorphism (SNP) differences between the donor and the recipient. Markers specific to donor SNPs are used to selectively amplify dd-cfDNA. However, this method is limited by the availability of donor information and the number of SNP differences between the donor and the recipient. 

  To mitigate this limitation, we've proposed a SNP-free approach of differentiating between dd-cfDNA and rd-cfDNA based on their tissue of origin. Since the majority of cell-free DNA in a healthy individual originate from blood cells, rd-cfDNA predominantly originates from blood cells whereas dd-cfDNA originats from lung cells. We've trained Deep Learning models to learn patterns indicative of the tissue of origin, to differentiate between dd-cfDNA and rd-cfDNA. 

### Explanation of feature sets 





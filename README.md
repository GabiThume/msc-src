
Description
-----------


1. __Image feature extraction, using quantization techniques and image descriptors.__


Quantization Techniques:

    Gleam
    Intensity
    Luminance
    MSB

Descriptors:

    BIC
    GCH
    CCV
    Haralick
    ACC

2. __Dimensionality reduction of feature vectors.__

Dimensionality Reduction Techniques:

    PCA
    Entropia

3. __Image classification.__

Classifier:

    Naive Bayes

4. __SMOTE: Synthetic Minority Over-sampling Technique.__

An implementation based on "SMOTE: Synthetic Minority Over-sampling Technique" paper's pseudo-code

5. __Artificial Generation of images to rebalance classes.__

Generate artificial images using preprocessing techniques to solve the balancing problem

Use
---

Before running the code, create a symbolic link to the images directory:

    ln -s <IMAGES DIRECTORY> BaseImagens

Makefile will compile the code for you:

    make

To run all descriptors and generate the feature vectors:

    ./runAllDescriptors.sh

After the previous command, to reduce the vectors dimension and apply the classification run:

    ./dimensionReduction <VECTORS DIRECTORY> <TECHNIQUE> <PARAMETERS LIST>

Options for techniques and parameters:

    [0] None:
        Just the classifier is going to be used, with the extracted vectors without dimensionality reduction.
    [1] PCA:
        - <nAttributes>: number of attributes to keep on PCA
    [2] Entropy:
        - <tWindow>: window size
    [3] All:
        - <nAttributes>: number of attributes to keep on PCA
        - <tWindow>: window size

The classification analysis (using Naive Bayes classifier and Repeated subsampling as a cross validation method) is going to be printed on the terminal. So, to write on a file the results of the analysis:

    ./dimensionReduction <VECTORS DIRECTORY> <TECHNIQUE> <PARAMETERS LIST>  >  analysis/<TECHNIQUE>_<PARAMETERS>.txt

Examples:

    ./dimensionReduction caracteristicas_corel/256/ 1 35 > analysis/Corel/PCA_50.txt
    ./dimensionReduction caracteristicas_corel/256/ 2 4 > analysis/Corel/ENTROPY_4.txt

Compare SMOTE and Artificial Generation of Images performance:

    ./rebalanceTest Desbalanced/original/ features/
    python plot.py # Check this file before running, it may modify a previous image

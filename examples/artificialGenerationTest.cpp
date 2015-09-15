/**
*
*	Author: Gabriela Thumé
*	Universidade de São Paulo / ICMC / 2014
*
**/

#include "preprocessing/artificialGeneration.h"

void classifica(string base, string features, string outfileName){

    Classifier c;
    int smallerClass, minoritySize;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir;
    stringstream id;
    Mat classes, trainTest;
    vector<Classes> data;

    /* Feature extraction */
    vector<int> parameters;
    descriptor(base.c_str(), features.c_str(), 4, 256, 1, 0, parameters, 0, 4, "");

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = readFeatures(name);
            if (data.size() != 0){

                smallerClass = c.findSmallerClass(data, &minoritySize);
                id << smallerClass;
                id << "_";
                id << minoritySize;

                c.classify(prob, 20, data, outfileName+id.str(), data[smallerClass].features.size().height);
            }
        }
    }
}

int main(int argc, char const *argv[]){

    Artificial a;

    if (argc != 3){
        cout << "\nUsage: ./artificialGenerationTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }

    cout << "\nRunning a rebalance test" << endl;

    /* Calculate before rebalance */
    classifica(string(argv[1]), string(argv[2]), "original_accuracy");

    /* */
    a.generate(string(argv[1]), string(argv[1])+"/../generated/", 0);

    /* Calculate after rebalance */
    classifica(string(argv[1]), string(argv[2]), "rebalance_accuracy");

    return 0;
}
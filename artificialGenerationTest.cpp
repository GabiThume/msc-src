/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "artificialGeneration.h"

void classifica(string base, string features, string outfileName){

    Classifier c;
    Size size;
    int numClasses, smallerClass, start, end;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir;
    stringstream id;
    Mat data, classes;
    pair <int, int> min(0,0);
    /* Feature extraction */
    descriptor(base.c_str(), features.c_str(), 4, 256, 1, 0, 0, 0, 0, 4, "");

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = readFeatures(name.c_str(), classes, numClasses);
            size = data.size();

            c.findSmallerClass(classes, numClasses, smallerClass, start, end);
            id << smallerClass;
            id << "_";
            id << end-start;

            if (size.height != 0){
                c.bayes(prob, 20, data, classes, numClasses, min, outfileName+id.str());
            }
        }
    }
}

int main(int argc, char const *argv[]){

    Artificial a;

    if (argc != 3){
        cout << "\nUsage: ./rebalanceTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }

    cout << "\nRunning a rebalance test" << endl;

    /* Calculate before rebalance */
    classifica(string(argv[1]), string(argv[2]), "original_accuracy");

    /* */
    a.generate(string(argv[1]), 1, 0);

    /* Calculate after rebalance */
    classifica(string(argv[1]), string(argv[2]), "rebalance_accuracy");

    return 0;
}
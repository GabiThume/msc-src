/**
*
*	Author: Gabriela Thumé
*	Universidade de São Paulo / ICMC / 2014
*
**/

#include "preprocessing/artificialGeneration.h"

void classifica(std::string base, std::string features, std::string outfileName){

    Classifier c;
    int smallerClass, minoritySize;
    double prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    std::ifstream myFile;
    std::string nameFile, name, nameDir;
    std::stringstream id;
    cv::Mat classes, trainTest;
    std::vector<ImageClass> data;

    /* Feature extraction */
    std::vector<int> parameters;
    PerformFeatureExtraction(base.c_str(), features.c_str(), 4, 256, 1, 0, parameters, 0, 4, "");

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = ReadFeaturesFromFile(name);
            if (data.size() != 0){

                smallerClass = c.findSmallerClass(data, &minoritySize);
                id << smallerClass;
                id << "_";
                id << minoritySize;

                c.classify(prob, 20, data, outfileName+id.str(), data[smallerClass].images.size());
            }
        }
    }
}

int main(int argc, char const *argv[]){

    Artificial a;

    if (argc != 3){
        std::cout << "\nUsage: ./artificialGenerationTest (1) (2)\n\n\t(1) Image Directory" << std::endl;
        std::cout << "\t(2) Features Directory\n" << std::endl;
        exit(-1);
    }

    std::cout << "\nRunning a rebalance test" << std::endl;

    /* Calculate before rebalance */
    classifica(std::string(argv[1]), std::string(argv[2]), "original_accuracy");

    /* */
    a.generate(std::string(argv[1]), std::string(argv[1])+"/../generated/", 0);

    /* Calculate after rebalance */
    classifica(std::string(argv[1]), std::string(argv[2]), "rebalance_accuracy");

    return 0;
}

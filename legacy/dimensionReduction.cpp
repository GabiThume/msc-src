/**
 * Functions for Dimensionality Reduction:
 *	PCA
 *	Entropy
 *
 * Classifier:
 *      Normal Bayes
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 **/

#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <map>
#include <cmath>
#include <vector>

#include "classifier.h"

double log2(double number){
   return log(number)/log(2);
}

cv::Mat entropyReduction(cv::Mat data, int tam_janela, std::string name_my_file){

    map<float, int> frequencias;
    map<float, int>::const_iterator iterator;
    double entropy, prob;
    int i, j, height, width, janela, fim_janela, indice_janela;
    std::string arq_saida;
    std::stringstream tam;
    tam << tam_janela;

    height = data.size().height;
    width = data.size().width;
    cv::Mat vectorEntropy(height, ceil((float)width/tam_janela), CV_32FC1);

    arq_saida = "entropy/ENTROPIA_" + tam.str() + "_"  + name_my_file;
    std::ofstream arq(arq_saida.c_str());

    for (i = 0; i < height; i++){
        janela = 0;
        indice_janela = 0;

        while (janela < width){
            entropy = 0;

            fim_janela = janela+tam_janela;
            if (fim_janela > width)
                fim_janela = width;

            frequencias.clear();
            for (j = janela; j < fim_janela; j++){
                float valor =  trunc(10000*data.at<float>(i, j))/10000;
                frequencias[valor]++;
            }

            for (iterator = frequencias.begin(); iterator != frequencias.end(); ++iterator) {
                prob = static_cast<double>(iterator->second) / (fim_janela -janela) ;
                entropy += prob * log2( prob ) ;
            }

            if (entropy != 0)
                entropy *= -1;
            vectorEntropy.at<float>(i, indice_janela) = (float)entropy;
            indice_janela++;
            janela += tam_janela;
        }
    }
    arq << vectorEntropy;
    arq.close();
    return vectorEntropy;
}

cv::Mat pcaReduction(cv::Mat data, int nComponents, std::string name_my_file){

    cv::Mat projection, eigenvectors;
    std::stringstream n;
    n << nComponents;

    std::string arq_saida = "pca/PCA_" + n.str() + "_" + name_my_file;
    std::ofstream arq(arq_saida.c_str());

    PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, nComponents);
    eigenvectors = pca.eigenvectors.clone();
    projection = pca.project(data);

    arq << projection;
    arq.close();
    return projection;
}

void inputError(){
    std::cout << "This programs waits: <directory> <technique> <attributes for PCA | window size for Entropy>\n";
    std::cout << "\tTechnique: 0-None, 1-PCA, 2-Entropy ou 3-All\n";
    exit(0);
}

int main(int argc, const char *argv[]){

    Classifier c;
    cv::Mat vectorEntropy, projection, classes, trainTest;
    std::vector<ImageClass> data;
    DIR *directory;
    struct dirent *arq;
    std::ifstream my_file;
    std::string name_arq, name_dir, name;
    int metodo, atributos, janela, i;
    pair <int, int> minority(-1,-1);
    float prob = 0.5;

    if (argc < 3)
        inputError();

	name_dir = argv[1];
	metodo = atoi(argv[2]);
    switch(metodo){
        case 0: /* Just classification */
            break;
        case 1: /* PCA */
            if (argc < 4)
                inputError();
            atributos = atoi(argv[3]);
            break;
        case 2: /* Entropy */
            if (argc < 4)
                inputError();
            janela = atoi(argv[3]);
            break;
        case 3: /* PCA + Entropy */
            if (argc < 5)
                inputError();
            atributos = atoi(argv[3]);
            janela = atoi(argv[4]);
            break;
        default:
            break;
    }

    /* For each file on input directory, do the following operations */
    directory = opendir(name_dir.c_str());
    if (directory != NULL){
       while ((arq = readdir(directory))){
            std::string out = "";
            name_arq = arq->d_name;
            name = name_dir + arq->d_name;
            my_file.open(name.c_str());

            int previousClass = -1;
            std::vector<ImageClass> dataInVector;
            Classes dataClass;
            cv::Mat dataMat, classesMat;

            if(my_file.good()){
                /* Read the feature vectors */
                data = readFeatures(name);
                if (data.size() != 0){

                    for(std::vector<ImageClass>::iterator it = data.begin(); it != data.end(); ++it) {
                        vconcat(dataMat, it->features, dataMat);
                        classesMat.resize(dataMat.size().height, it->classNumber);
                    }

                    switch(metodo){
                        case 0:
                            std::cout << std::endl << "Classification for "<< name.c_str() << std::endl;
                            c.classify(prob, 10, data, out.c_str(), 0);
                            break;
                        case 1:
                            std::cout << std::endl << "PCA for "<< name.c_str() << " with " << atributos << " attributes" << std::endl;
                            projection = pcaReduction(dataMat, atributos, name_arq);
                            for (i = 0; i < projection.size().height; ++i){
                                if (previousClass != classesMat.at<float>(i,0)){
                                    if (previousClass != -1){
                                        dataInVector.push_back(dataClass);
                                    }
                                    dataClass.features.create(1, projection.size().width, CV_32FC1);
                                }

                                dataClass.features.resize(dataClass.features.size().height+1);
                                dataClass.features.row(dataClass.features.size().height) = projection.row(i);
                            }

                            c.classify(prob, 10, dataInVector, out.c_str(), atributos);
                            break;
                        case 2:
                            std::cout << std::endl << "Entropy for "<< name.c_str() << " with window = " << janela << std::endl;
                            vectorEntropy = entropyReduction(dataMat, janela, name_arq);

                            for (i = 0; i < vectorEntropy.size().height; ++i){
                                if (previousClass != classesMat.at<float>(i,0)){
                                    if (previousClass != -1){
                                        dataInVector.push_back(dataClass);
                                    }
                                    dataClass.features.create(1, projection.size().width, CV_32FC1);
                                }

                                dataClass.features.resize(dataClass.features.size().height+1);
                                dataClass.features.row(dataClass.features.size().height) = projection.row(i);
                            }

                            c.classify(prob, 10, dataInVector, out.c_str(), janela);
                            break;
                        case 3:
                            std::cout << std::endl << "Classification for "<< name.c_str() << std::endl;
                            c.classify(prob, 10, data, out.c_str(), 0);

                            std::cout << std::endl << "PCA for "<< name.c_str() << " with " << atributos << " attributes" << std::endl;
                            projection = pcaReduction(dataMat, atributos, name_arq);
                            for (i = 0; i < projection.size().height; ++i){
                                if (previousClass != classesMat.at<float>(i,0)){
                                    if (previousClass != -1){
                                        dataInVector.push_back(dataClass);
                                    }
                                    dataClass.features.create(1, projection.size().width, CV_32FC1);
                                }

                                dataClass.features.resize(dataClass.features.size().height+1);
                                dataClass.features.row(dataClass.features.size().height) = projection.row(i);
                            }
                            c.classify(prob, 10, dataInVector, out.c_str(), atributos);

                            std::cout << std::endl << "Entropy for "<< name.c_str() << " with window = " << janela << std::endl;
                            vectorEntropy = entropyReduction(dataMat, janela, name_arq);
                            for (i = 0; i < vectorEntropy.size().height; ++i){
                                if (previousClass != classesMat.at<float>(i,0)){
                                    if (previousClass != -1){
                                        dataInVector.push_back(dataClass);
                                    }
                                    dataClass.features.create(1, projection.size().width, CV_32FC1);
                                }

                                dataClass.features.resize(dataClass.features.size().height+1);
                                dataClass.features.row(dataClass.features.size().height) = projection.row(i);
                            }
                            c.classify(prob, 10, dataInVector, out.c_str(), janela);
                            break;
                        default:
                            break;
                    }
                }
            }
            my_file.close();
       }
    }
    return 0;
}

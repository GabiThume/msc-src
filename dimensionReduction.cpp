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

#include "funcoesArquivo.h"
#include "classifier.h"

using namespace cv;
using namespace std;

double log2(double number){
   return log(number)/log(2);
}

Mat entropyReduction(Mat data, int tam_janela, string nome_my_file){

    map<float, int> frequencias;
    map<float, int>::const_iterator iterator;
    double entropy, prob;
    int i, j, height, width, janela, fim_janela, indice_janela;
    string arq_saida;
    stringstream tam;
    tam << tam_janela;
        
    height = data.size().height;
    width = data.size().width;
    Mat vectorEntropy(height, ceil((float)width/tam_janela), CV_32FC1);

    arq_saida = "entropy/ENTROPIA_" + tam.str() + "_"  + nome_my_file; 
    ofstream arq(arq_saida.c_str());
   
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

Mat pcaReduction(Mat data, int nComponents, string nome_my_file){

    Mat projecao, autovectores;
    stringstream n;
    n << nComponents;

    string arq_saida = "pca/PCA_" + n.str() + "_" + nome_my_file; 
    ofstream arq(arq_saida.c_str());

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, nComponents);
    autovectores = pca.eigenvectors.clone();
    projecao = pca.project(data);

    arq << projecao;
    arq.close();    
    return projecao;
}

void inputError(){
    cout << "This programs waits: <directory> <technique> <attributes for PCA | window size for Entropy>\n";
    cout << "\tTechnique: 0-None, 1-PCA, 2-Entropy ou 3-All\n";
    exit(0);
}

int main(int argc, const char *argv[]){

    Classifier c;
    Mat vectorEntropy, projecao, data, classes;
    DIR *directory;
    struct dirent *arq;
    ifstream my_file;
    string nome_arq, nome_dir, nome;
    int nClasses, metodo, atributos, janela;
    pair <int, int> minority(0,0);
    float prob = 0.5;

    if (argc < 3)
        inputError();

	nome_dir = argv[1];
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
    directory = opendir(nome_dir.c_str());
    if (directory != NULL){
       while ((arq = readdir(directory))){
            string out = "";
            nome_arq = arq->d_name;
            nome = nome_dir + arq->d_name;
            my_file.open(nome.c_str());

            if(my_file.good()){
                /* Read the feature vectors */
                data = readFeatures(nome.c_str(), classes, nClasses);
                if (data.size().height != 0){

                    switch(metodo){
                        case 0:
                            cout << endl << "Classification for "<< nome.c_str() << endl;
                            c.bayes(prob, 10, data, classes, nClasses, minority, out.c_str());
                            break;
                        case 1:
                            cout << endl << "PCA for "<< nome.c_str() << " with " << atributos << " attributes" << endl;
                            projecao = pcaReduction(data, atributos, nome_arq);
                            c.bayes(prob, 10, projecao, classes, nClasses, minority, out.c_str());
                            break;
                        case 2:
                            cout << endl << "Entropy for "<< nome.c_str() << " with window = " << janela << endl;
                            vectorEntropy = entropyReduction(data, janela, nome_arq);
                            c.bayes(prob, 10, vectorEntropy, classes, nClasses, minority, out.c_str());
                            break;
                        case 3:
                            cout << endl << "Classification for "<< nome.c_str() << endl;
                            c.bayes(prob, 10, data, classes, nClasses, minority, out.c_str());
                            cout << endl << "PCA for "<< nome.c_str() << " with " << atributos << " attributes" << endl;
                            projecao = pcaReduction(data, atributos, nome_arq);
                            c.bayes(prob, 10, projecao, classes, nClasses, minority, out.c_str());
                            cout << endl << "Entropy for "<< nome.c_str() << " with window = " << janela << endl;
                            vectorEntropy = entropyReduction(data, janela, nome_arq);
                            c.bayes(prob, 10, vectorEntropy, classes, nClasses, minority, out.c_str());
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


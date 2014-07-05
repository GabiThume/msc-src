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

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <map>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

Mat readFeatures(const string& filename, vector<int> &classes, int *nClasses) {

    string linha, infos, numero_img, classe, num_caract, num_classes, objetos;
    vector<float> vec;
    float valor, caract;
    size_t n, d;
    int i, j;
    ifstream my_file(filename.c_str());
    if(!my_file)
        throw exception();

    // Read the first line, which contains the number of objects, classes and features
    getline(my_file, infos);
    if (infos == "")
        return Mat();
    stringstream info(infos);
    getline(info, objetos, '\t');
    getline(info, num_classes, '\t');
    (*nClasses) = atoi(num_classes.c_str());
    getline(info, num_caract, '\t');

    // Create a Mat with the file data
    n = atoi(objetos.c_str());
    d = atoi(num_caract.c_str());
    Mat data(n, d, CV_32FC1);

    while (getline(my_file, linha)) {

        stringstream vector_caract(linha);
        getline(vector_caract, numero_img, '\t');
        getline(vector_caract, classe, '\t');
        i = atoi(numero_img.c_str());
        j = 0;
        while(vector_caract >> caract) {
            data.at<float>(i, j) = (float)caract;
            j++;
        }
        classes.push_back(atoi(classe.c_str()));
    }    

    my_file.close();
    return data;
}

int bayesClassifier(Mat vectorFeatures, vector<int> classes, int num_classes, float prob) {

    Mat resultados;
    CvNormalBayesClassifier classificador;
    int i, j, acertos, total, height, width, treinados, classe_atual;
    int iTreino, iTesting, num_componentes, num_treino, num_Testing, totalTreino;
    int conjunto_treino, inicio, data_classe, pos, repetition, num_repetition;
    float deviation_standard, variance, media;
    vector<float> acuracy;
    vector<int> vector_rand;
    srand(time(0)); 
    num_repetition = 10;
        
    Size n = vectorFeatures.size();
    height = n.height;
    width = n.width;
    num_treino = ceil(height*prob);
    num_Testing = height-num_treino;
    
    Mat dataTraining(num_treino, width, CV_32FC1);
    Mat rotulosTraining(num_treino, 1, CV_32FC1);
    Mat dataTesting(num_Testing, width, CV_32FC1);
    Mat rotulosTesting(num_Testing, 1, CV_32FC1);

    /* Repeated random sub-sampling validation */

    // For each repetition, divide randomly the set in training and testing
    conjunto_treino = num_treino/num_classes;

    // This supose the classes are in balanced number - Simplification
    data_classe = height/num_classes;

    for(repetition = 0; repetition < num_repetition; repetition++) {

        iTreino = 0; treinados = 0;

        for (i = 0; i < num_classes; i++) {

            treinados = 0;
            classe_atual = i+1;
            inicio = (classe_atual-1)*data_classe;
            while (treinados < conjunto_treino) {
                // Generate a random position for a training data
                pos = inicio + (rand() % (data_classe));
                if (!count(vector_rand.begin(), vector_rand.end(), pos)){
                    vector_rand.push_back(pos);
                    Mat treino = dataTraining.row(iTreino); 
                    vectorFeatures.row(pos).copyTo(treino);
                    rotulosTraining.at<float>(iTreino, 0) = classes[pos];
                    treinados++;
                    iTreino++;
               }
            }
        }  

	// After selecting the training set, the testing set it is going to be the rest of the whole set
        iTesting = 0;
        for (i = 0; i < height; i++) {
            if (!count(vector_rand.begin(), vector_rand.end(), i)){
                Mat Testing = dataTesting.row(iTesting);
                vectorFeatures.row(i).copyTo(Testing);
                rotulosTesting.at<float>(iTesting, 0) = classes[i];
                iTesting++;
            }
        }
        vector_rand.clear();

        // Train and predict using the Normal Bayes classifier    
        classificador.train(dataTraining, rotulosTraining);
        classificador.predict(dataTesting, &resultados);
     
        acertos = 0;
        for (i = 0; i < resultados.size().height; i++) {
            if (rotulosTesting.at<float>(i, 0) == resultados.at<float>(i, 0)) {
                acertos++;        
            }
        }
            
        total = resultados.size().height;
        totalTreino = rotulosTraining.size().height;
        acuracy.push_back(acertos*100.0/total);
    }

    media = acuracy[0];
    for (i = 1; i < acuracy.size(); i++){
        media = (media+acuracy[i])/2;
    }

    variance = 0;
    for (i = 0; i < acuracy.size(); i++){
        variance += pow(acuracy[i]-media, 2);
    }
    variance = variance/acuracy.size();
    deviation_standard = sqrt(variance);

    cout << "Cross validation for set " << total << " Testings and " << totalTreino << " Training with "<<acuracy.size()<<" repetitions:" << endl;
    cout << "\tMean = " << media << endl;
    cout << "\tVariance = " << variance << endl;
    cout << "\tStandard Deviation = " << deviation_standard << endl << endl; 

    dataTraining.release();
    dataTesting.release();
    rotulosTesting.release();
    rotulosTraining.release();
    return 0;
}

double log2(double number) {
   return log(number) / log(2) ;
}

Mat entropyReduction(Mat data, int tam_janela, string nome_my_file) {

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

Mat pcaReduction(Mat data, int nComponents, string nome_my_file) {

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

int main(int argc, const char *argv[]) {

    Mat vectorEntropy, projecao, data;
    vector<int> classes;
    DIR *directory;
    struct dirent *arq;
    ifstream my_file;
    string nome_arq, nome_dir, nome;
    int nClasses, metodo, atributos, janela;

    if (argc < 3)
        inputError();
	nome_dir = argv[1];
	metodo = atoi(argv[2]);

    switch(metodo){
        case 0: // Just classification
            break;
        case 1: // PCA
            if (argc < 4) 
                inputError();
            atributos = atoi(argv[3]);
            break;
        case 2: // Entropy
            if (argc < 4) 
                inputError();
            janela = atoi(argv[3]);
            break;
        case 3: // PCA + Entropy
            if (argc < 5) 
                inputError();
            atributos = atoi(argv[3]);
            janela = atoi(argv[4]);
            break;
        default:
            break;
    }
    
    float prob = 0.5;

    // For each file on input directory, do the following operations	    
    directory = opendir(nome_dir.c_str());
    if (directory != NULL){
       while (arq = readdir(directory)){

            nome_arq = arq->d_name;
            nome = nome_dir + arq->d_name;
            my_file.open(nome.c_str());

            if(my_file.good()){
                // Read the feature vectors
                data = readFeatures(nome.c_str(), classes, &nClasses);
                if (data.size().height != 0){

                    switch(metodo){
                        case 0: // Just classification
                            cout << endl << "Classification for "<< nome.c_str() << endl;
                            bayesClassifier(data, classes, nClasses, prob);
                            break;
                        case 1: // PCA
                            cout << endl << "PCA for "<< nome.c_str() << " com " << atributos << " atributos" << endl;
                            projecao = pcaReduction(data, atributos, nome_arq);
                            bayesClassifier(projecao, classes, nClasses, prob);
                            break;
                        case 2: // Entropy
                            cout << endl << "Entropy for "<< nome.c_str() << " com janela = " << janela << endl;
                            vectorEntropy = entropyReduction(data, janela, nome_arq);
                            bayesClassifier(vectorEntropy, classes, nClasses, prob);
                            break;
                        case 3: // PCA, Entropy
                            cout << endl << "Classification for "<< nome.c_str() << endl;
                            bayesClassifier(data, classes, nClasses, prob);
                            cout << endl << "PCA for "<< nome.c_str() << " com " << atributos << " atributos" << endl;
                            projecao = pcaReduction(data, atributos, nome_arq);
                            bayesClassifier(projecao, classes, nClasses, prob);
                            cout << endl << "Entropy for "<< nome.c_str() << " com janela = " << janela << endl;
                            vectorEntropy = entropyReduction(data, janela, nome_arq);
                            bayesClassifier(vectorEntropy, classes, nClasses, prob);
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


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

Mat leituraCaracteristicas(const string& filename, vector<int> &classes){
    string linha, infos, numero_img, classe, num_caract, num_classes, objetos;
    vector<float> vec;
    float valor, caract;
    size_t n, d;
    int i, j;
         
    ifstream arquivo(filename.c_str());
    if(!arquivo)
        throw exception();

    // Leitura da primeira linha que contém a quantidade de objetos
    // o número de classes e o número de características
    getline(arquivo, infos);
    if (infos == "")
        return Mat();
    stringstream info(infos);
    getline(info, objetos, '\t');
    getline(info, num_classes, '\t');
    getline(info, num_caract, '\t');

    // Cria uma Mat com os dados do arquivo
    n = atoi(objetos.c_str());
    d = atoi(num_caract.c_str());
    Mat data(n, d, CV_32FC1);

    while (getline(arquivo, linha)) {

        stringstream vetor_caract(linha);

        getline(vetor_caract, numero_img, '\t');
        getline(vetor_caract, classe, '\t');
        i = atoi(numero_img.c_str());
        j = 0;

        while(vetor_caract >> caract) {
            data.at<float>(i, j) = (float)caract;
            j++;
        }
        classes.push_back(atoi(classe.c_str()));
    }    
    arquivo.close();
    return data;
}

int classificacaoBayes(Mat vetorCaracteristicas, vector<int> classes, float p){
    Mat resultados;
    CvNormalBayesClassifier classificador;
    int i, j, acertos, total, height, width, treinados, classeAnterior;
    int iTreino, iTeste, num_componentes, num_treino, num_teste, totalTreino;
    int num_classes;
    float acuracia;
    
    cout << "vet = " << classes << endl;
    
    Size n = vetorCaracteristicas.size();
    height = n.height;
    width = n.width;
    num_treino = (int)(height*p);
    num_teste = (int)(height*(1.0-p));
    num_classes = 10;
    // Armazena os dados de treinamento e de teste para classificação
    Mat dadosTreinamento(num_treino, width, CV_32FC1);
    Mat rotulosTreinamento(num_treino, 1, CV_32FC1);
    Mat dadosTeste(num_teste, width, CV_32FC1);
    Mat rotulosTeste(num_teste, 1, CV_32FC1);

    classeAnterior = -1; iTreino = 0; iTeste = 0;

    for (i = 0; i < height; i++) {  

        if (classes[i] != classeAnterior){
            treinados = 0;
            classeAnterior = classes[i];
        }

        if (treinados < num_treino/num_classes) {
            for (j = 0; j < width; j++) {
                dadosTreinamento.at<float>(iTreino, j) = vetorCaracteristicas.at<float>(i, j);  
            }  
            rotulosTreinamento.at<float>(iTreino, 0) = classes[i];
            treinados++;
            iTreino++;
        } else {
            for (j = 0; j < width; j++) {  
                dadosTeste.at<float>(iTeste, j) = vetorCaracteristicas.at<float>(i, j);  
            }  
            rotulosTeste.at<float>(iTeste, 0) = classes[i];
            iTeste++;
        }
    }  

    // Treina o classificador Normal Bayes    
    classificador.train(dadosTreinamento, rotulosTreinamento);


    // Avalia o classificador Normal Bayes
    classificador.predict(dadosTeste, &resultados);

    //cout <<"Rotulos = " << rotulosTeste << endl << endl;  
    //cout <<"Classificação = " << resultados << endl;  
 
    acertos = 0;
    for (i = 0; i < resultados.size().height; i++) {
        if (rotulosTeste.at<float>(i, 0) == resultados.at<float>(i, 0)) {
            acertos++;        
        }
    }
    
    total = resultados.size().height;
    totalTreino = rotulosTreinamento.size().height;
    acuracia = acertos*100.0/total;
    cout << "Acurácia: " << acuracia << " Para a execução de: " << total << " testes e " << totalTreino << " treinos." << endl;

    return 0;
}

double log2(double number) {
   return log(number) / log(2) ;
}

Mat calculaEntropia(Mat data, int tam_janela){

    map<float , int> frequencias;
    map<float, int>::const_iterator iterator;
    double entropia, freq;
    int i, j, height, width, janela, fimJanela;
        
    height = data.size().height;
    width = data.size().width;
    Mat vetorEntropia(height, ceil(width/tam_janela), CV_32FC1);
    
    for (i = 0; i < height; i++){
        janela = 0;
        int iJanela = 0;

        while (janela < width){
            entropia = 0;
            frequencias.clear();

            fimJanela = janela+tam_janela;
            if (fimJanela > width)
                fimJanela = width;
            
            for (j = janela; j < fimJanela; j++) {
                frequencias[data.at<float>(i, j)]++;
            }

            for (iterator = frequencias.begin(); iterator != frequencias.end(); ++iterator) {
                freq = static_cast<double>(iterator->second) / width ;
                entropia += freq * log2( freq ) ;
            }
            entropia *= -1;
            //cout << "Entropia = " << entropia << endl;

            vetorEntropia.at<float>(i,iJanela) = (float)entropia;
            iJanela++;

            janela += tam_janela;
        }

    }
    return vetorEntropia;
}

Mat calculaPCA(Mat data, int nComponents){
    Mat projecao, autovetores;

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, nComponents);
    autovetores = pca.eigenvectors.clone();
    projecao = pca.project(data);

    return projecao;
}

//  ./<executavel> <diretorio> <tipo_execucao> [lista_parametros]
//  Opcoes:
//      PCA: usa o pca
//          - nAtributos: numero de atributos da projecao
//      Entropy: usa entropia
//          - tJanela: tamanho da janela
//      ALL: roda todos os metodos
//          - nAtributos: numero de atributos da projecao
//          - tJanela: tamanho da janela
//      Caso nao seja passado parametro somente o Naive Bayes sera usado

int main(int argc, const char *argv[]){
    Mat vetorEntropia, projecao, data;
    vector<int> classes;
    DIR *diretorio;
    struct dirent *arq;
    ifstream arquivo;
    string nome_arq, nome_dir;

    nome_dir = "caracteristicas/";
    diretorio = opendir(nome_dir.c_str());

    if (diretorio != NULL){
       while (arq = readdir(diretorio)){
            nome_arq = nome_dir + arq->d_name;
            arquivo.open(nome_arq.c_str());

            if(arquivo.good()){

                data = leituraCaracteristicas(nome_arq.c_str(), classes);
                if (data.size().height != 0){

                    cout << endl << nome_arq << endl;

                    projecao = calculaPCA(data, 35);
                    cout << "dsdada" << endl;
                    classificacaoBayes(projecao, classes, 0.7);
                    //vetorEntropia = calculaEntropia(data, 4);
                    //classificacaoBayes(vetorEntropia, classes);

                }
            }
            arquivo.close();
       }
    }
    return 0;
}

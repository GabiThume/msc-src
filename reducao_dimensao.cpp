#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>

using namespace cv;
using namespace std;

Mat leituraCaracteristicas(const string& filename, vector<int> &classes) 
{
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

int classificacaoBayes(Mat projecao, vector<int> classes){

    Mat resultados;
    CvNormalBayesClassifier classificador;
    int i, j, acertos, total, height, width, treinados, classeAnterior;
    int iTreino, iTeste, num_componentes, num_treino, num_teste, totalTreino;
    float acuracia;
    
    Size n = projecao.size();
    height = n.height;
    width = n.width;
    num_treino = (int)(height*0.2);
    num_teste = (int)(height*0.8);
    
    if (height == 0) {
        return 0;
    }
    
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

        if (treinados < 20) {
            for (j = 0; j < width; j++) {
                dadosTreinamento.at<float>(iTreino, j) = projecao.at<float>(i, j);  
            }  
            rotulosTreinamento.at<float>(iTreino, 0) = classes[i];
            treinados++;
            iTreino++;
        } else {
            for (j = 0; j < width; j++) {  
                dadosTeste.at<float>(iTeste, j) = projecao.at<float>(i, j);  
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

Mat realizaPCA(string nome_arq, vector<int>& classes){

    Mat projecao, autovetores, data;
    int num_componentes;

    data = leituraCaracteristicas(nome_arq.c_str(), classes);
    if (data.size().height != 0){
        cout << endl << nome_arq.c_str() << endl;
     
        num_componentes = 10;
        PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_componentes);
        autovetores = pca.eigenvectors.clone();
        projecao = pca.project(data);
    }
    return projecao;
}


int main(int argc, const char *argv[]) 
{
    Mat resultados, autovetores, projecao, data;
    vector<int> classes;

    //data = leituraCaracteristicas("caracteristicas/BaseImagens_Haralick6_Luminance_256c_100r.txt", classes);

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
                
                projecao = realizaPCA(nome_arq, classes);
                classificacaoBayes(projecao, classes);
                //cout << nome_arq << endl;
            }
            arquivo.close();
       }
    }

    
    
    return 0;
}

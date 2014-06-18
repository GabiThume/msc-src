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

Mat leituraCaracteristicas(const string& filename, vector<int> &classes, int *nClasses) {

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
    (*nClasses) = atoi(num_classes.c_str());
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

int classificacaoBayes(Mat vetorCaracteristicas, vector<int> classes, int num_classes, float prob) {

    Mat resultados;
    CvNormalBayesClassifier classificador;
    int i, j, acertos, total, height, width, treinados, classeAnterior;
    int iTreino, iTeste, num_componentes, num_treino, num_teste, totalTreino;
    float acuracia;
        
    Size n = vetorCaracteristicas.size();
    height = n.height;
    width = n.width;
    num_treino = (int)round(height*prob);
    num_teste = (int)round(height*(1.0-prob));
    
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

Mat calculaEntropia(Mat data, int tam_janela, string nome_arquivo) {

    map<float, int> frequencias;
    map<float, int>::const_iterator iterator;
    double entropia, freq;
    int i, j, height, width, janela, fim_janela, indice_janela;
    string arq_saida;
    stringstream tam;
    tam << tam_janela;
        
    height = data.size().height;
    width = data.size().width;
    Mat vetorEntropia(height, ceil((float)width/tam_janela), CV_32FC1);

    arq_saida = "entropia/ENTROPIA_" + tam.str() + "_"  + nome_arquivo; 
    ofstream arq(arq_saida.c_str());
   
    for (i = 0; i < height; i++){
        janela = 0;
        indice_janela = 0;
        
        while (janela < width){
            entropia = 0;

            fim_janela = janela+tam_janela;
            if (fim_janela > width)
                fim_janela = width;
            
            frequencias.clear();
            for (j = janela; j < fim_janela; j++){
                float valor =  trunc(1000*data.at<float>(i, j))/1000;
                frequencias[valor]++;
            }
            
            for (iterator = frequencias.begin(); iterator != frequencias.end(); ++iterator) {
                freq = static_cast<double>(iterator->second) / (fim_janela -janela) ;
                entropia += freq * log2( freq ) ;
            }

            entropia *= -1;
            vetorEntropia.at<float>(i, indice_janela) = (float)entropia;
            indice_janela++;
            janela += tam_janela;
        }
    }
    arq << vetorEntropia;
    cout << endl << arq_saida << endl;
    arq.close();    
    return vetorEntropia;
}

Mat calculaPCA(Mat data, int nComponents, string nome_arquivo) {

    Mat projecao, autovetores;
    stringstream n;
    n << nComponents;

    string arq_saida = "pca/PCA_" + n.str() + "_" + nome_arquivo; 
    ofstream arq(arq_saida.c_str());

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, nComponents);
    autovetores = pca.eigenvectors.clone();
    projecao = pca.project(data);

    arq << projecao;
    cout << endl << arq_saida << endl;
    arq.close();    

    return projecao;
}

void erroEntrada(){
    cout << "Esse programa espera: <diretorio> <metodo> <atributos para o PCA | tamanho da janela para Entropia>\n";
    cout << "\tMétodo: 0-Nenhum, 1-PCA, 2-Entropia ou 3-Todos\n";
    exit(0);
}

int main(int argc, const char *argv[]) {

    Mat vetorEntropia, projecao, data;
    vector<int> classes;
    DIR *diretorio;
    struct dirent *arq;
    ifstream arquivo;
    string nome_arq, nome_dir, nome;
    int nClasses, metodo, atributos, janela;

    if (argc < 3)
        erroEntrada();

	nome_dir = argv[1];

	metodo = atoi(argv[2]); // descritor a ser utilizado
    switch(metodo){
    case 0: // Somente classificação
        break;
    case 1: // PCA
        if (argc < 4) 
            erroEntrada();
        atributos = atoi(argv[3]);
        break;
    case 2: // Entropia
        if (argc < 4) 
            erroEntrada();
        janela = atoi(argv[3]);
        break;
    case 3: // PCA + Entropia
        if (argc < 5) 
            erroEntrada();
        atributos = atoi(argv[3]);
        janela = atoi(argv[4]);
        break;
    default:
        break;
    }

    // Para cada arquivo no diretório de entrada, realiza as operações	    
    diretorio = opendir(nome_dir.c_str());
    if (diretorio != NULL){
       while (arq = readdir(diretorio)){

            nome_arq = arq->d_name;
            nome = nome_dir + arq->d_name;
            arquivo.open(nome.c_str());

            if(arquivo.good()){
                // Lê os dados dos vetores de características do arquivo
                data = leituraCaracteristicas(nome.c_str(), classes, &nClasses);
                if (data.size().height != 0){

                    //cout << endl << nome_arq << endl;

                    switch(metodo){
                    case 0: // Somente classificação
                        classificacaoBayes(data, classes, nClasses, 0.2);
                        break;
                    case 1: // PCA
                        projecao = calculaPCA(data, atributos, nome_arq);
                        classificacaoBayes(projecao, classes, nClasses, 0.2);
                        break;
                    case 2: // Entropia
                        vetorEntropia = calculaEntropia(data, janela, nome_arq);
                        classificacaoBayes(vetorEntropia, classes, nClasses, 0.2);
                        break;
                    case 3: // PCA + Entropia
                        projecao = calculaPCA(data, atributos, nome_arq);
                        classificacaoBayes(projecao, classes, nClasses, 0.2);
                        vetorEntropia = calculaEntropia(data, janela, nome_arq);
                        classificacaoBayes(vetorEntropia, classes, nClasses, 0.2);
                        break;
                    default:
                        break;
                    }
                }
            }
            arquivo.close();
       }
    }
    return 0;
}


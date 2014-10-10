#include "funcoesArquivo.h"

int main(int argc, char *argv[]){

    string id, base, dir;
    int descritor, nColor, nRes, oNorm, quantMethod, oZero, totalpar;

    if (argc < 8){
        cout << "O programa espera: <pasta> <pasta_descritor> <descritor> <quantidade de cores para quantizar a imagem> ";
        cout << "fator de redimensionamento> <normalizacao> <metodo quantizacao> <distancias ACC | limiar CCV> ";
        cout << "id para arquivo de saida, caso necessário>\n";
        cout << " - Descritores: 1 - BIC   2- GCH   3- CCV     4- Haralick    5- AutoCorrelograma (ACC)\n";
        cout << " - Cores: 8, 16, 32, 64 ou 256\n";
        cout << " - Redimensionamento - positivo, com máximo = 1 (1 = 100%)\n";
        cout << " - Normalizacao - 0 (sem normalizacao) 1 (entre 0 e 1), 2 (0 a 255)\n";
        //cout << " - Descartar colunas nulas (atributos nulos) - 1 descartar, 0 nao descartar\n";
        cout << " - Sequencia de distancias para ACC ou limiar do CCV\n";
        exit(0);
    } 

    if (argc == 9){
        id = argv[8];
    }
    else{
        id = "";
    }

	base = argv[1];
	dir = argv[2];
	descritor = atoi(argv[3]);
	if((descritor < 1) || (descritor > 5)) {
	    cout << "Descritor nao existe!!\n\n";
	    return -1;
	}
	nColor = atoi(argv[4]);
	nRes = atof(argv[5]);
	oNorm = atoi(argv[6]);
	quantMethod = atoi(argv[7]);
 	// oZero = atoi(argv[6]); - remover colunas
	oZero = 0;
	
	totalpar = (argc-8);
	int *params = new int[totalpar];
	if (descritor == 3) {
	    params[0] = atoi(argv[6]);
	}
	else if (descritor == 5){
        for (int i = 0; i < totalpar; i++){
            params[i] = atoi(argv[6+i]);
        }
	}
	
	if((nColor != 8) && (nColor != 16) && (nColor != 32) && (nColor != 64) && (nColor != 128) && (nColor != 256)){
	    cout << "Quantidade de cores deve ser 8, 16, 32, 64, 128 ou 256!!\n\n";
	    return -1;
	}
	if((nRes <= 0) || (nRes > 1)){
	    cout << "Redimensionamento deve ser positivo e menor ou igual a 1\n\n";
	    return -1;
	}
	if((oNorm < 0) || (oNorm > 2)){
	    cout << "Normalizacao invalida (0, 1 ou 2)\n\n";
	    return -1;
	}
	if((oZero != 0) && (oZero != 1)){
	    cout << "Opcao descartar invalida (0, 1)\n\n";
	    return -1;
	}
	
	descriptor(base.c_str(), dir.c_str(), descritor, nColor, nRes, oNorm, params, totalpar, oZero, quantMethod, id.c_str());
	
	return 1;
}
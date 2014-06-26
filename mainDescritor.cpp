#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;

#include "funcoesArquivo.h"

int main(int argc, char *argv[])
{
    if (argc < 8){
        cout << "O programa espera: <pasta> <pasta_descritor> <descritor> <quantidade de cores para quantizar a imagem> <fator de redimensionamento> <normalizacao> <metodo quantizacao> <distancias ACC | limiar CCV>\n";
        cout << " - Descritores: 1 - BIC   2- GCH   3- CCV     4- Haralick    5- AutoCorrelograma (ACC)\n";
        cout << " - Cores: 8, 16, 32, 64 ou 256\n";
        cout << " - Redimensionamento - positivo, com máximo = 1 (1 = 100%)\n";
        cout << " - Normalizacao - 0 (sem normalizacao) 1 (entre 0 e 1), 2 (0 a 255)\n";
        //cout << " - Descartar colunas nulas (atributos nulos) - 1 descartar, 0 nao descartar\n";
        cout << " - Sequencia de distancias para ACC ou limiar do CCV\n";
        exit(0);
    } 

	const char *base = argv[1]; // pasta com base de imagens
	const char *dir = argv[2]; 
	int descritor = atoi(argv[3]); // descritor a ser utilizado
	int nColor = atoi(argv[4]); // quantidade de cores
	double nRes = atof(argv[5]); // alteracao na resolucao
	int oNorm = atoi(argv[6]); // normalizacao
	int quantMethod = atoi(argv[7]);
// 	int oZero = atoi(argv[6]); // remover colunas
	int oZero = 0;
	
	if( (descritor < 1) || (descritor > 5) ) {
	    cout << "Descritor nao existe!!\n\n"; 
	    return -1; 
	}
	
	int totalpar = (argc-8);
	
	int *params = new int[totalpar];
	
	if (descritor == 3) {
	    params[0] = atoi(argv[6]);
	} else if (descritor == 5) {
        for (int i = 0; i < totalpar; i++)
            params[i] = atoi(argv[6+i]);
	}
	
	if( (nColor != 8) && (nColor != 16) && (nColor != 32) && (nColor != 64) && (nColor != 128) && (nColor != 256) ) 
		{ cout << "Quantidade de cores deve ser 8, 16, 32, 64, 128 ou 256!!\n\n"; return -1; }
	
	if( (nRes <= 0) || (nRes > 1) ) { cout << "Redimensionamento deve ser positivo e menor ou igual a 1\n\n"; return -1; }
	
	if( (oNorm < 0) || (oNorm > 2) ) { cout << "Normalizacao invalida (0, 1 ou 2)\n\n"; return -1; }

	if( (oZero != 0) && (oZero != 1) ) { cout << "Opcao descartar invalida (0, 1)\n\n"; return -1; }
	
	descriptor(base, dir, descritor, nColor, nRes, oNorm, params, totalpar, oZero, quantMethod);
	
	return 1;
}

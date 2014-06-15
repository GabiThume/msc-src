#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;

#include "funcoesArquivo.h"

/**
 * 	Exemplos
 * 	./mainDescritor BaseImagens 5 64 1 1 3 9
 *         - pasta "BaseImagens", descritor ACC, imagens com tamanho 100%, normalizacao entre 0 e 1 e conjunto de distancias {3 e 9}
 *
 * 	./mainDescritor BaseImagens 4 8 0.5 0
 *         - pasta "BaseImagens", descritor Haralick, imagens com tamanho 50%, sem normalizacao
 **/


int main(int argc, char *argv[])
{
    if (argc < 7) 
    {
	 cout << "O programa espera: <pasta> <descritor> <quantidade de cores para quantizar a imagem> <fator de redimensionamento> <normalizacao> <metodo quantizacao> <distancias ACC | limiar CCV>\n";
	 cout << " - Descritores: 1 - BIC   2- GCH   3- CCV     4- Haralick    5- AutoCorrelograma (ACC)\n";
	 cout << " - Cores: 8, 16, 32, 64 ou 256\n";
	 cout << " - Redimensionamento - positivo, com máximo = 1 (1 = 100%)\n";
	 cout << " - Normalizacao - 0 (sem normalizacao) 1 (entre 0 e 1), 2 (0 a 255)\n";
	 //cout << " - Descartar colunas nulas (atributos nulos) - 1 descartar, 0 nao descartar\n";
	 cout << " - Sequencia de distancias para ACC ou limiar do CCV\n";
	 exit(0);
    } 

	const char *base = argv[1]; // pasta com base de imagens
	int descritor = atoi(argv[2]); // descritor a ser utilizado
	int nColor = atoi(argv[3]); // quantidade de cores
	double nRes = atof(argv[4]); // alteracao na resolucao
	int oNorm = atoi(argv[5]); // normalizacao
	int quantMethod = atoi(argv[6]);
// 	int oZero = atoi(argv[6]); // remover colunas
	int oZero = 0; // remover colunas
	
	if( (descritor < 1) || (descritor > 5) ) { cout << "Descritor nao existe!!\n\n"; return -1; }
	
	int totalpar = (argc-7);
	
	int *params = new int[totalpar];
	
	if (descritor == 3) {
	    params[0] = atoi(argv[7]);
	} else if (descritor == 5) {
	    for (int i = 0; i < totalpar; i++) {
		params[i] = atoi(argv[7+i]);
	    }
	}
	
	if( (nColor != 8) && (nColor != 16) && (nColor != 32) && (nColor != 64) && (nColor != 128) && (nColor != 256) ) 
		{ cout << "Quantidade de cores deve ser 8, 16, 32, 64, 128 ou 256!!\n\n"; return -1; }
	
	if( (nRes <= 0) || (nRes > 1) ) { cout << "Redimensionamento deve ser positivo e menor ou igual a 1\n\n"; return -1; }
	
	if( (oNorm < 0) || (oNorm > 2) ) { cout << "Normalizacao invalida (0, 1 ou 2)\n\n"; return -1; }

	if( (oZero != 0) && (oZero != 1) ) { cout << "Opcao descartar invalida (0, 1)\n\n"; return -1; }
	
	descriptor(base, descritor, nColor, nRes, oNorm, params, totalpar, oZero, quantMethod);
	
	return 1;
}

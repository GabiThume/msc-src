/*
	Luciana Calixta Escobar
	
	
	Biblioteca que contem as seguintes funcoes auxiliares:
	- QuantizationMSB
	- distManhattan
	- distEuclid
	- distChessboard
*/


#ifndef _FUNCOESAUX_H
#define _FUNCOESAUX_H

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;


/*	Funcao QuantizationMSB
 Q uantiza uma imagem de acordo com a quantidade *de cores passada por argumento
 Requer:
 - imagem a ser quantizada
 - imagem onde sera armazenada a imagem quantizada
 - quantidade de cores 
 Retorna: 
 - quantidade unicas de cores da imagem quantizada */
int QuantizationMSB(Mat &I, Mat &Q, int nColors);


void RemoveNullColumns(Mat &Feat);

/* Normaliza um histograma
 * Funcao para normalizar (entre 0 e 255) o histograma gerado pelo descritor BIC
 * Requer:
 *	- o histograma a ser normalizado
 *	- um histograma ja alocado, para guardar o resultado
 *	- o tamanho do vetor
 *	- fator de normalizacao */
void NormalizeHist(long int *hist, float *histnorm, int nColor, int fator);


/*	Funcao distManhattan
	Calcula a diferenca entre dois histrogramas atraves da distancia Manhattan
	Requer:
		- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
		- tamanho do histograma 
	Retorna:
		- a distancia entre os dois histogramas */
double distManhattan(double *p, double *q, int size);


/*	Funcao distEuclid
	Calcula a diferenca entre dois histrogramas atraves da distancia Euclidiana
	Requer:
		- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
		- tamanho do histograma 
	Retorna:
		- a distancia entre os dois histogramas */
double distEuclid(double *q, double *p, int size);


/*	Funcao distChessboard
	Calcula a diferenca entre dois histrogramas atraves da distancia Chessboard
	Requer:
		- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
		- tamanho do histograma 
	Retorna:
		- a distancia entre os dois histogramas */
double distChessboard(double *p, double *q, int size);


#endif


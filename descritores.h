/*
	Luciana Calixta Escobar
	
	
	Biblioteca que contem 
	
*/


#ifndef _DESCRITORES_H
#define _DESCRITORES_H

#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <queue>
#include <vector>

#include "funcoesAux.h"

using namespace cv;
using namespace std;

typedef struct
{
	int i;
	int j;
	uchar color;
} Pixel;




/* Funcao Find Neighbor
 * Encontra os vizinhos de um pixel
 * Usada no descritor CCV
 * Requer:
 *	- imagem original
 *	- fila de pixels
 *	- pixels ja visitados
 *	- tamanho da regiao */
void find_neighbor(Mat & img, queue<Pixel> & pixels, int * visited, long int & tam_reg);


/* Descritor CCV
 * Cria dois histogramas de cor da imagem:
 * 1 -> histograma de pixels coerentes
 * 2 -> histograma de pixels incoerentes
 * Requer:
 *	- imagem original
 *	- fistograma ja alocado, com duas vezes a quantidade de cor
 *	- quantidade de cores usadas na imagem
 *	- nivel de coerencia */	
void CCV(Mat & img, Mat &features, int nColor, int oNorm, int threshold);


/* Descritor GCH
 * Cria histrograma de cor da imagem.
 * Requer:
 *	- imagem original
 *	- histrograma ja alocado
 *	- quantidade de cores usadas na imagem */
void GCH(Mat &I, Mat &features, int nColor, int oNorm);


/* Descritor BIC
 * Cria dois histrogramas de cor da imagem:
 * 1 -> histograma de borda
 * 2 -> histograma de interior
 * Requer:
 *	- imagem original
 *	- histograma ja alocado, com tamanho de duas vezes a quantidade de cor
 *	- quantidade de cores usadas na imagem 
 * No histograma, de 0 até (nColor -1) = Borda, de nColor até (2*nColor -1) = Interior */
void BIC(Mat &I, Mat &features, int nColor, int oNorm);


/* Funcao CoocurrenceMatrix
 * Cria uma matriz que contem a ocorrencia de cada cor em cada pixel da imagem
 * Requer:
 *	- imagem original
 *	- matriz ja alocada, no tamanho nColor x nColor
 *	- quantidade de cores usadas na imagem
 *	- coordenadas dX e dY, que podem ser 0 ou 1 */
void CoocurrenceMatrix(Mat &I, double **Cm, int nColor, int dX, int dY);


/* Funcao Haralick6
 * Cria um histograma com 6 descritores de textura
 * Requer:
 *	- matriz de coocorrencia
 *	- quantidade de cores usadas na imagem
 *	- histograma ja alocado
 * Os descritores sao:
 *	- maxima Probabilidade
 *	- correlacao
 *	- contraste
 *	- energia (uniformidade)
 *	- homogeneidade 
 *	- entropia */
void Haralick6(double **Cm, int nColor, Mat &features);


/* Descritor Haralick
 * Cria um histograma com 6 descritores de textura
 * Chama as funcoes Haralick6 e CoocurrenceMatrix
 * Requer:
 * 	- imagem original
 *	- matriz de coocorrencia
 *	- quantidade de cores usadas na imagem
 *	- histograma ja alocado
 * Os descritores sao:
 *	- maxima Probabilidade
 *	- correlacao
 *	- contraste
 *	- energia (uniformidade)
 *	- homogeneidade 
 *	- entropia */
void HARALICK(Mat &I, double **Cm, Mat &features, int nColor, int oNorm);


/* Descritor Autocorrelograma
 * Cria um histograma de cor que descreve a distribuição 
 * global da correlação entre a localização espacial de cores
 * Requer:
 *	- imagem original
 *	- valor da distancia k entre os pixels
 *	- histograma ja alocado
 *	- quantidade de cores usadas na imagem */
void ACC(Mat &I, Mat &features, int nColor, int oNorm, int *k, int totalk);


#endif

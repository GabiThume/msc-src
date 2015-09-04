/*
Copyright (c) 2015, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Gabriela Thumé nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors:  Gabriela Thumé (gabithume@gmail.com)
          Moacir Antonelli Ponti (moacirponti@gmail.com)
Universidade de São Paulo / ICMC
Master's thesis in Computer Science
*/

#ifndef _DESCRITORES_H
#define _DESCRITORES_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include "funcoesAux.h"
#include <opencv2/nonfree/nonfree.hpp>

// #include "vlfeat/vl/sift.h"
#include <vl/sift.h>

using namespace cv;
using namespace std;

const string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};
const string descriptorMethod[9] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"};

typedef struct {
    int i;
    int j;
    uchar color;
} Pixel;

/* Funcao Find Neighbor
 * Encontra os vizinhos de um pixel
 * Usada no descritor CCV
 * Requer:
 *    - imagem original
 *    - fila de pixels
 *    - pixels ja visitados
 *    - tamanho da regiao */
void find_neighbor(Mat img, queue<Pixel> *pixels, int *visited, long int *tam_reg);


/* Descritor CCV
 * Cria dois histogramas de cor da imagem:
 * 1 -> histograma de pixels coerentes
 * 2 -> histograma de pixels incoerentes
 * Requer:
 *    - imagem original
 *    - fistograma ja alocado, com duas vezes a quantidade de cor
 *    - quantidade de cores usadas na imagem
 *    - nivel de coerencia */
void CCV(Mat img, Mat *features, int nColor, int oNorm, int threshold);


/* Descritor GCH
 * Cria histrograma de cor da imagem.
 * Requer:
 *    - imagem original
 *    - histrograma ja alocado
 *    - quantidade de cores usadas na imagem */
void GCH(Mat I, Mat *features, int nColor, int oNorm);


/* Descritor BIC
 * Cria dois histrogramas de cor da imagem:
 * 1 -> histograma de borda
 * 2 -> histograma de interior
 * Requer:
 *    - imagem original
 *    - histograma ja alocado, com tamanho de duas vezes a quantidade de cor
 *    - quantidade de cores usadas na imagem
 * No histograma, de 0 até (nColor -1) = Borda, de nColor até (2*nColor -1) = Interior */
void BIC(Mat I, Mat *features, int nColor, int oNorm);


/* Funcao CoocurrenceMatrix
 * Cria uma matriz que contem a ocorrencia de cada cor em cada pixel da imagem
 * Requer:
 *    - imagem original
 *    - matriz ja alocada, no tamanho nColor x nColor
 *    - quantidade de cores usadas na imagem
 *    - coordenadas dX e dY, que podem ser 0 ou 1 */
void CoocurrenceMatrix(Mat I, double **Cm, int nColor, int dX, int dY);


/* Funcao Haralick6
 * Cria um histograma com 6 descritores de textura
 * Requer:
 *    - matriz de coocorrencia
 *    - quantidade de cores usadas na imagem
 *    - histograma ja alocado
 * Os descritores sao:
 *    - maxima Probabilidade
 *    - correlacao
 *    - contraste
 *    - energia (uniformidade)
 *    - homogeneidade
 *    - entropia */
void Haralick6(double **Cm, int nColor, Mat *features);


/* Descritor Haralick
 * Cria um histograma com 6 descritores de textura
 * Chama as funcoes Haralick6 e CoocurrenceMatrix
 * Requer:
 *     - imagem original
 *    - matriz de coocorrencia
 *    - quantidade de cores usadas na imagem
 *    - histograma ja alocado
 * Os descritores sao:
 *    - maxima Probabilidade
 *    - correlacao
 *    - contraste
 *    - energia (uniformidade)
 *    - homogeneidade
 *    - entropia */
void HARALICK(Mat I, Mat *features, int nColor, int oNorm);


/* Descritor Autocorrelograma
 * Cria um histograma de cor que descreve a distribuição
 * global da correlação entre a localização espacial de cores
 * Requer:
 *    - imagem original
 *    - valor da distancia k entre os pixels
 *    - histograma ja alocado
 *    - quantidade de cores usadas na imagem */
void ACC(Mat I, Mat *features, int nColor, int oNorm, int *k, int totalk);

void LBP(Mat img, Mat *features, int colors);

void HOG(Mat img, Mat *features, int numFeatures);

void contourExtraction(Mat Img, Mat *features);

// void surf(Mat Img, Mat *features);

#endif

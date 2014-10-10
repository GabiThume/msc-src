#ifndef _FUNCOES_ARQUIVO_H
#define _FUNCOES_ARQUIVO_H

#include <stdlib.h>
#include <string.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "descritores.h"
#include "funcoesAux.h"

using namespace cv;
using namespace std;


Mat readFeatures(const string& filename, Mat &classes, int &nClasses);

int qtdArquivos(char *diretorio);

int qtdImagensTotal(const char * base, int qtdClasses);

int descriptor(const char *baseImagem, char const *diretorioDescritores, int method, int nColor, double nRes, int oNorm, int *param, int nparam, int oZero, int quantMethod, char const *id);

#endif

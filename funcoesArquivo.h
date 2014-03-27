#ifndef _FUNCOES_ARQUIVO_H
#define _FUNCOES_ARQUIVO_H

#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include "descritores.h"


using namespace cv;
using namespace std;


int qtdArquivos(char *diretorio);

int qtdImagensTotal(const char * base, int qtdClasses);

int descriptor(const char *baseImagem, int method, int nColor, double nRes, int oNorm, int *param, int nparam, int oZero);

#endif

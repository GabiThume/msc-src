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


Mat readFeatures(string filename, Mat *classes, int *nClasses);

int qtdArquivos(string diretorio);

int qtdImagensTotal(string base, int qtdClasses);

int descriptor(string baseImagem, string diretorioDescritores, int method, int nColor, double nRes, int oNorm, int *param, int nparam, int oZero, int quantMethod, string id);

#endif

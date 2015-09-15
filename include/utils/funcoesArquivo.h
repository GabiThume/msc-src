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
#include "description/descritores.h"
#include "utils/funcoesAux.h"
#include "quantization/quantization.h"

using namespace cv;
using namespace std;

struct Classes {
	int classNumber;
	Mat features;
	bool fixedTrainOrTest;
	Mat trainOrTest;
};

vector<Classes> readFeatures(string filename);

int qtdArquivos(string diretorio);

int qtdImagensTotal(string base, int qtdClasses, vector<int> *objClass, int *maxs);

string descriptor(string baseImagem, string diretorioDescritores, int method, int nColor, double nRes, int oNorm, vector<int> param, int oZero, int quantMethod, string id);

#endif
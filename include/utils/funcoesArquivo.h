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

vector<Classes> ReadFeaturesFromFile(string filename);

int qtdArquivos(string diretorio);

int NumberImagesInDataset(string base, int qtdClasses, vector<int> *objClass);

string PerformFeatureExtraction(string database, string featuresDir, int method,
    int colors, double resizeFactor, int normalization, vector<int> param,
    int deleteNull, int quantization, string id = "");

#endif

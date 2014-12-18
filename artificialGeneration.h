#ifndef _ARTIFICIAL_H
#define _ARTIFICIAL_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

#include "funcoesArquivo.h"
#include "classifier.h"

using namespace cv;
using namespace std;

class Artificial{

    public:
        int generate(string base, int isToGenerate);
};


#endif

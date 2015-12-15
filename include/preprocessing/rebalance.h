#ifndef _REBALANCE_H
#define _REBALANCE_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

#include "utils/data.h"
#include "classification/classifier.h"

using namespace cv;
using namespace std;


void generate();

class Rebalance{

    public:
        void generate();
};


#endif

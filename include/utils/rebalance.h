#ifndef _REBALANCE_H
#define _REBALANCE_H

// #include "utils/dataStructure.h"
#include "preprocessing/smote.h"
#include "description/descritores.h"
#include "quantization/quantization.h"
#include "classification/classifier.h"

#include <dirent.h>

class Rebalance {

    public:
      int colors;
      int normalization;

      Data data;
      std::string imagesDirectory, featuresDirectory, analysisDirectory;

      FeatureExtraction extractor;
      GrayscaleConversion quantization;

      void readImageDirectory(std::string directory);
      void shuffleImages(std::vector<Image> *img);
      void separateInFolds(int k);
      std::vector<std::vector<double> > classify(std::string descriptorFile,
        int repeat, double prob, std::string csv);
      void performFeatureExtraction(int extractMethod, int grayMethod);
      std::string performSmote(Data imbalancedData, int operation);
      void writeFeatures(std::string id);
};

int qtdArquivos(std::string directory);

#endif

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
      void ShuffleImages(std::vector<Image> *img);
      void SeparateInFolds(std::vector<ImageClass> *original_data, int k);
      std::vector<std::vector<double> > classify(std::string descriptorFile,
        int repeat, double prob, std::string csv);
      void performFeatureExtraction(int extractMethod, int grayMethod);
      std::string PerformSmote(Data imbalancedData, int operation);
      void writeFeatures(std::string id);
};

int qtdArquivos(std::string directory);
int NumberImagesInDataset(std::string base, int qtdClasses,
  std::vector<int> *objClass);
cv::Mat FindImgInClass(std::string database, int img_class, int img_number,
  int index, int treino, cv::Mat *trainTest, std::vector<std::string> *path, cv::Mat *isGenerated);
void NumberImgInClass(std::string database, int img_class, int *num_imgs,
  int *num_train);

#endif

#ifndef _REBALANCE_H
#define _REBALANCE_H

#include "utils/data.h"
#include "preprocessing/smote.h"

using namespace cv;
using namespace std;

class Rebalance {

    public:
      Data thisData;

      FeatureExtractor extractor(colors, normalization);
      extractor.ccvThreshold = 25;
      extractor.accDistances = {1, 3, 5, 7};
      extractor.colors = 64;
      extractor.normalization = 0;
      extractor.resizeFactor = 1.0;

      GrayscaleConversion quantization(colors);

};

string Rebalance::PerformSmote(vector<ImageClass> imbalancedData, int operation, string csvSmote);
string Rebalance::RemoveSamples(string database, string newDir, double prob, double id);
void Rebalance::PerformFeatureExtractionFromData(int method, int colors,
  double resizeFactor, int normalization, vector<int> param, int deleteNull,
  int quantization);
vector<vector<double> > Rebalance::classify(string descriptorFile, int repeat, double prob, string csv);

#endif

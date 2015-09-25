/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalanceTest.h"

string desc(string dir, string features, int d, int m, string id){

    vector<int> paramCCV = {25};
    vector<int> paramACC = {1, 3, 5, 7};
    vector<int> parameters;
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramCCV, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramACC, 0, m, id);
    else
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, parameters, 0, m, id);
}

int main(int argc, char const *argv[]){

  Classifier c;
  int numClasses, m, d, indexDescriptor, minoritySize;
  double prob = 0.5;
  string newDir, baseDir, featuresDir, csvOriginal, directory, str, bestDir;
  vector<Classes> originalData;

  if (argc != 3){
    cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
    cout << "\t(0) Directory to place tests\n" << endl;
    cout << "\t(1) Image Directory\n" << endl;
    cout << "\t(2) Features Directory\n" << endl;
    cout << "\t(3) Analysis Directory\n" << endl;
    cout << "\t./rebalanceTest Desbalanced/ Desbalanced/original/ Desbalanced/features/ Desbalanced/analysis/ 0\n" << endl;
    exit(-1);
  }
  newDir = string(argv[1]);
  baseDir = string(argv[2]);
  // featuresDir = string(argv[3]);
  // analysisDir = string(argv[4]);

  /*  Available
  descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
  Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
  */
  vector <int> descriptors {1, 2, 3, 4, 5, 6, 7, 8};

  for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
    d = descriptors[indexDescriptor];
    for (m = 1; m <= 5; m++){
      csvOriginal = newDir+"/analysis/original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      featuresDir = newDir+"/features/";

      /* Feature extraction from images */
      string originalDescriptor = desc(baseDir, featuresDir, d, m, "original");
      /* Read the feature vectors */
      originalData = ReadFeaturesFromFile(originalDescriptor);
      numClasses = originalData.size();
      if (numClasses != 0){
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using original data" << endl;
        cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        c.findSmallerClass(originalData, &minoritySize);
        c.classify(prob, 1, originalData, csvOriginal.c_str(), minoritySize);
        originalData.clear();
      }
    }
  }
  return 0;
}

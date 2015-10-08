#ifndef _REBALANCE_H
#define _REBALANCE_H

#include "utils/funcoesArquivo.h"
#include "preprocessing/smote.h"

using namespace cv;
using namespace std;

// class Rebalance {
//
//     public:
//
//       string PerformSmote(vector<Classes> imbalancedData, int operation, string csvSmote);
//       string RemoveSamples(string database, string newDir, double prob, double id);
// };

string PerformSmote(vector<Classes> imbalancedData, int operation, string csvSmote);
string RemoveSamples(string database, string newDir, double prob, double id);

#endif

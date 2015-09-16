/*
Copyright (c) 2015, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Gabriela Thumé nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors:  Gabriela Thumé (gabithume@gmail.com)
          Moacir Antonelli Ponti (moacirponti@gmail.com)
Universidade de São Paulo / ICMC
Master's thesis in Computer Science
*/

#include <string>
#include <vector>
#include "utils/funcoesArquivo.h"

int main(int argc, char *argv[]) {
  int descMethod, colors, normalization, quantMethod, deleteNull, numParameters;
  vector<int> params;
  double resize;
  string id = "", databaseDir = "", featuresDir = "";

  if (argc < 9) {
    cout << "\nUsage: ./descriptor (0) (1) (2) (3) (4) (5) (6) (7) (8) (9)\n\n";
    cout << "\t(0) Images Directory\n" << endl;
    cout << "\t(1) Features Directory\n" << endl;
    cout << "\t(2) Descriptor Method:\n" << endl;
    cout << "\t\t1-BIC  2-GCH  3-CCV  4-Haralick  5-ACC  6-LBP  7-HOG ";
    cout << "8-Contour  9-Fisher\n" << endl;
    cout << "\t(3) Number of Colors:\n" << endl;
    cout << "\t\t8, 16, 32, 64 or 256\n";
    cout << "\t(4) Resize factor:\n" << endl;
    cout << "\t\t[0-1] (1 = 100%)\n";
    cout << "\t(5) Normalization:\n" << endl;
    cout << "\t\t0-without 1-[0-1] interval 2-[0-255] interval\n";
    cout << "\t(6) Quantization Method:\n" << endl;
    cout << "\t\t1-Intensity  2-Luminance  3-Gleam  4-MSB\n" << endl;
    cout << "\t(7) Null columns:\n" << endl;
    cout << "\t\t1-Remove 0-Keep\n" << endl;
    cout << "\t(8) Id to identify outfile\n" << endl;
    cout << "\t(9) Distances for ACC or CCV threshold\n" << endl;
    exit(-1);
  }

  databaseDir = argv[1];
  featuresDir = argv[2];

  descMethod = atoi(argv[3]);
  if (descMethod < 1 || descMethod > 10) {
    cout << "This descriptor method does not exist.\n";
    exit(-1);
  }

  colors = atoi(argv[4]);
  if (colors != 8 && colors != 16 && colors != 32 && colors != 64 &&
    colors != 128 && colors != 256) {
      cout << "The number of colors must be 8, 16, 32, 64, 128 or 256." << endl;
      exit(-1);
    }

    resize = atof(argv[5]);
    if (resize <= 0 || resize > 1) {
      cout << "The resize factor must be between 0 and 1." << endl;
      exit(-1);
    }

    normalization = atoi(argv[6]);
    if (normalization != 0 && normalization != 1 && normalization != 2) {
      cout << "Invalid normalization (use 0, 1 or 2)." << endl;
      exit(-1);
    }

    quantMethod = atoi(argv[7]);

    deleteNull = atoi(argv[8]);
    if (deleteNull != 0 && deleteNull != 1) {
      cout << "Wrong delete option (0, 1)." << endl;
      exit(-1);
    }

    if (argc >= 9) {
      id = argv[9];
    }

    if (descMethod == 3 || descMethod == 5) {
      numParameters = (argc-10);
      for (int i = 0; i < numParameters; ++i) {
        params.push_back(atoi(argv[10+i]));
      }
    }

    descriptor(databaseDir, featuresDir, descMethod, colors, resize,
              normalization, params, deleteNull, quantMethod, id);
    return 0;
  }

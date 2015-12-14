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

#include "utils/rebalance.h"

/*******************************************************************************
Generate a imbalanced class
Requires:
- vector<Classes>
- int
- string
*******************************************************************************/
string PerformSmote(vector<Classes> imbalancedData, int operation, string csvSmote) {

  int majority = -1, majorityClass = -1, eachClass, amountSmote, countImg = 0;
  int i, x, neighbors, pos, total, h, w;
  int samples, index;
  vector<int> trainingNumber(imbalancedData.size(), 0);
  std::vector<Classes>::iterator it;
  vector<Classes> rebalancedData;
  stringstream numberOfImages;
  ofstream arq;
  Mat synthetic;
  SMOTE s;

  for (it = imbalancedData.begin(); it != imbalancedData.end(); ++it){
    if (it->fixedTrainOrTest){
      for(i = 0; i < it->images.size(); ++i){
        if (it->images[i].isFreeTrainOrTest == 1)
          trainingNumber[it->classNumber]++;
      }
    }
    else {
      trainingNumber[it->classNumber] = it->images.size()/2.0;
    }
    if (trainingNumber[it->classNumber] > majority){
      majorityClass = it->classNumber;
      majority = trainingNumber[it->classNumber];
    }
  }

  total = 0;
  for (eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass) {

    Mat dataTraining(0, imbalancedData[eachClass].images[0].features.size(), CV_32FC1);
    Mat dataTesting(0, imbalancedData[eachClass].images[0].features.size(), CV_32FC1);
    Mat dataRaw(0, imbalancedData[eachClass].images[0].features.size(), CV_32FC1);

    cout << "In class " << eachClass << " were found " << trainingNumber[eachClass] << " original training images"<< endl;
    /* Find out how many samples are needed to rebalance */
    amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];
    cout << "Amount to smote: " << amountSmote << " samples" << endl;
    if (amountSmote > 0){
      //neighbors = 5;
      neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
      cout << "Number of neighbors: " << neighbors << endl;
      cout << "imbalancedData[eachClass].images.size() " << imbalancedData[eachClass].images.size() << endl;
      for (x = 0; x < imbalancedData[eachClass].images.size(); ++x){
        if (imbalancedData[eachClass].images[x].isFreeTrainOrTest == 1){
          dataTraining.push_back(imbalancedData[eachClass].images[x].features);
        } else if (imbalancedData[eachClass].images[x].isFreeTrainOrTest == 2){
          dataTesting.push_back(imbalancedData[eachClass].images[x].features);
        } else {
          dataRaw.push_back(imbalancedData[eachClass].images[x].features);
        }
      }

      cout << dataTraining.rows << " " << dataTesting.rows << endl;
      if (dataTraining.rows > 0 && dataTesting.rows > 0) {
        if (operation != 0){
          synthetic = s.smote(dataTraining, amountSmote, neighbors);
        } else {
          synthetic.create(amountSmote, imbalancedData[eachClass].images[0].features.size(), CV_32FC1);
          for (x = 0; x < amountSmote; x++){
            pos = rand() % (dataTraining.size().height);
            Mat tmp = synthetic.row(x);
            dataTraining.row(pos).copyTo(tmp);
          }
        }

        cout << "SMOTE generated " << amountSmote << " new synthetic samples" << endl;

        /* Concatenate original with synthetic data*/
        Classes imgClass;
        Image img;
        imgClass.classNumber = eachClass;

        // Training
        for (samples = 0; samples < dataTraining.rows; samples++) {
          dataTraining.row(samples).copyTo(img.features);
          img.isFreeTrainOrTest = 1;
          img.isGenerated = 0;
          img.path = imbalancedData[eachClass].images[samples].path;
          imgClass.images.push_back(img);
        }
        // New synthethic examples
        for (index = 0; index < synthetic.rows; index++) {
          synthetic.row(index).copyTo(img.features);
          img.isFreeTrainOrTest = 2;
          img.isGenerated = 1;
          img.path = "smote";
          imgClass.images.push_back(img);
        }
        // Testing
        for (index = 0; index < dataTesting.rows; samples++, index++) {
          dataTesting.row(index).copyTo(img.features);
          img.isFreeTrainOrTest = 2;
          img.isGenerated = 1;
          img.path = imbalancedData[eachClass].images[samples].path;
          imgClass.images.push_back(img);
        }

        rebalancedData.push_back(imgClass);
        total += imgClass.images.size();
        dataTraining.release();
        synthetic.release();
        dataTesting.release();
      } else if (dataRaw.rows > 0) {
        // Mat data(0, imbalancedData[eachClass].images[0].features.size(), CV_32FC1);
        // for (samples = 0; samples < imbalancedData[eachClass].images.size(); samples++) {
        //   data.push_back(imbalancedData[eachClass].images[samples].features);
        // }

        synthetic = s.smote(dataRaw, amountSmote, neighbors);
        cout << "SMOTE generated " << synthetic.rows << " new synthetic samples" << endl;
        Classes imgClass;
        Image img;

        imgClass.classNumber = eachClass;
        for (samples = 0; samples < imbalancedData[eachClass].images.size(); samples++) {
          img.features = imbalancedData[eachClass].images[samples].features;
          img.isFreeTrainOrTest = 0;
          img.isGenerated = 0;
          img.path = imbalancedData[eachClass].images[samples].path;
          imgClass.images.push_back(img);
        }
        for (samples = 0; samples < synthetic.rows; samples++) {
          synthetic.row(samples).copyTo(img.features);
          img.isFreeTrainOrTest = 0;
          img.isGenerated = 1;
          img.path = "smote";
          imgClass.images.push_back(img);
        }

        rebalancedData.push_back(imgClass);
        total += imgClass.images.size();
      }
    } else {
      rebalancedData.push_back(imbalancedData[eachClass]);
      total += imbalancedData[eachClass].images.size();
    }
  }

  numberOfImages.str("");
  numberOfImages << total;

  string name = csvSmote + numberOfImages.str()+"i_smote.csv";
  arq.open(name.c_str(), ios::out);
  if (!arq.is_open()) {
    cout << "It is not possible to open the feature's file: " << name << endl;
    exit(-1);
  }
  arq << total << ',' << rebalancedData.size() << ',' << rebalancedData[0].images[0].features.size() << endl;
  for(std::vector<Classes>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
    for (h = 0; h < it->images.size(); h++){
      arq << it->images[h].path << ',' << it->classNumber << ',' << it->images[h].isFreeTrainOrTest << ',';
      arq << it->images[h].isGenerated << ',';
      for (w = 0; w < it->images[0].features.cols-1; w++){
        arq << it->images[h].features[w] << ",";
      }
      arq << it->images[h].features[w] << endl;
      countImg++;
    }
  }
  arq.close();
  cout << "---------------------------------------------------------------------------------------" << endl;
  cout << "Wrote on data file named " << name << endl;
  cout << "---------------------------------------------------------------------------------------" << endl;
  return name;
}

/*******************************************************************************
Generate a imbalanced class
Requires:
- string
- string
- double
- double
*******************************************************************************/
string RemoveSamples(string database, string newDir, double prob, double id) {

  int pos = 0, samples, imagesTraining, i, imgsInClass, x, qtdClasses;
  int prev, count_diff, treino, num_images, imagesTesting;
  vector<int> vectorRand, objperClass;
  string str, nameFile, name, nameDir, directory, dir;
  stringstream numImages, classNumber, image, globalFactor;
  Size size;
  ifstream myFile;
  Mat data, classes;
  double fator = 1.0;

  // Check how many classes and images there are
  directory = database+"/";
  qtdClasses = qtdArquivos(directory);
  if (qtdClasses < 2) {
    cout << "Error. There is less than two classes in " << directory << endl;
    exit(-1);
  }
  NumberImgInClass(directory, 0, &prev, &treino);
  imagesTesting = prev;
  count_diff = 0;
  for (i = 1; i < qtdClasses; i++) {
    NumberImgInClass(database, i, &num_images, &treino);
    if (num_images < imagesTesting) imagesTesting = num_images;
    count_diff = count_diff + abs(num_images - prev);
    prev = num_images;
  }
  imagesTesting = imagesTesting*prob;
  if (count_diff == 0) {
    cout << "\n\n------------------------------------------------------------------------------------" << endl;
    cout << "Divide the number of original samples to create a minority class:" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
  }

  globalFactor << id;
  dir = newDir+"/Imbalance-"+globalFactor.str()+"/";
  str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
  str += "cp -r "+database+"* "+dir+";";
  system(str.c_str());

  for(i = 0; i < qtdClasses; i++) {
    classNumber.str("");
    classNumber << i;
    directory = database + "/" + classNumber.str()  + "/";
    imgsInClass = qtdArquivos(directory);
    if (imgsInClass == 0){
      fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
      exit(-1);
    }
    samples = ceil((1.0 + fator)*imgsInClass*prob);
    if (samples > 0){

      numImages.str("");
      numImages << vectorRand.size();

      //str = "rm "+dir+classNumber.str()+"/*;";
      str = "rm -f -r "+dir+classNumber.str()+"/*;";
      str += "mkdir -p "+dir+classNumber.str()+"/treino/;";
      str += "mkdir -p "+dir+classNumber.str()+"/teste/;";

      //cout << " Executa " << str.c_str() << endl;
      system(str.c_str());

      /* Generate a random position to select samples to create the minority class */
      while ((int)vectorRand.size() < samples) {
        pos = rand() % imgsInClass;
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
          vectorRand.push_back(pos);
        }
      }

      /* Copy some of the originals to a training folder */
      // imagesTraining = vectorRand.size()*prob;
      imagesTraining = vectorRand.size() - imagesTesting;
      for(x = 0; x < imagesTraining; x++){
        image.str("");
        image << vectorRand[x];
        str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
        str+= dir+classNumber.str()+"/treino/; ";
        system(str.c_str());
        //cout << " Executa " << str.c_str() << endl;
      }

      /* Copy the rest of originals to a testing folder */
      for(x = imagesTraining; x < (int)vectorRand.size(); x++){
        image.str("");
        image << vectorRand[x];
        str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
        str+= dir+classNumber.str()+"/teste/;";
        system(str.c_str());
        //cout << " Executa " << str.c_str() << endl;
      }

      str = "bash scripts/rename.sh "+dir+classNumber.str()+"/";
      system(str.c_str());

      vectorRand.clear();
    }
    if (count_diff == 0) {
      // fator -= id/(double)qtdClasses;
      if (id != 1.0) {
        fator -= (1.0 - id);
      } else {
        fator -= 1.0/(double)qtdClasses;
      }
    }
  }
  return dir;
}

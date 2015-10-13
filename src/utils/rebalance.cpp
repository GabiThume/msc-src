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
  int numTraining = 0, numTesting = 0, i, x, neighbors, pos, total, h, w;
  vector<int> trainingNumber(imbalancedData.size(), 0);
  std::vector<Classes>::iterator it;
  vector<Classes> rebalancedData;
  stringstream numberOfImages;
  ofstream arq;
  Mat synthetic;
  SMOTE s;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "SMOTE generation of samples to rebalance classes" << endl;
  cout << "---------------------------------------------------------" << endl;

  for (it = imbalancedData.begin(); it != imbalancedData.end(); ++it){
    if (it->fixedTrainOrTest){
      for(i = 0; i < it->trainOrTest.size().height; ++i){
        if (it->trainOrTest.at<int>(i,0) == 1)
          trainingNumber[it->classNumber]++;
      }
    }
    else {
      trainingNumber[it->classNumber] = it->trainOrTest.size().height/2.0;
    }
    if (trainingNumber[it->classNumber] > majority){
      majorityClass = it->classNumber;
      majority = trainingNumber[it->classNumber];
    }
  }

  total = 0;
  for (eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass) {

    Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_32FC1);
    Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_32FC1);

    numTraining = 0;
    numTesting = 0;

    cout << "In class " << eachClass << " were found " << trainingNumber[eachClass] << " original training images"<< endl;
    /* Find out how many samples are needed to rebalance */
    amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];
    cout << "Amount to smote: " << amountSmote << " samples" << endl;
    if (amountSmote > 0){
      //neighbors = 5;
      neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
      cout << "Number of neighbors: " << neighbors << endl;

      for (x = 0; x < imbalancedData[eachClass].trainOrTest.size().height; ++x){
        if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 1){
          dataTraining.resize(numTraining+1);
          Mat tmp = dataTraining.row(numTraining);
          imbalancedData[eachClass].features.row(x).copyTo(tmp);
          numTraining++;
        } else if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 2){
          dataTesting.resize(numTesting+1);
          Mat tmp = dataTesting.row(numTesting);
          imbalancedData[eachClass].features.row(x).copyTo(tmp);
          numTesting++;
        }
      }

      if (dataTraining.rows > 0 && dataTesting.rows > 0) {
        if (operation != 0){
          synthetic = s.smote(dataTraining, amountSmote, neighbors);
        } else {
          cout << imbalancedData[eachClass].features.size().width << endl;
          synthetic.create(amountSmote, imbalancedData[eachClass].features.size().width, CV_32FC1);
          for (x = 0; x < amountSmote; x++){
            pos = rand() % (dataTraining.size().height);
            Mat tmp = synthetic.row(x);
            dataTraining.row(pos).copyTo(tmp);
          }
        }

        cout << "SMOTE generated " << amountSmote << " new synthetic samples" << endl;

        /* Concatenate original with synthetic data*/
        Classes imgClass;
        Mat dataRebalanced;
        vconcat(dataTraining, synthetic, dataRebalanced);
        vconcat(dataRebalanced, dataTesting, imgClass.features);
        imgClass.classNumber = eachClass;
        imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32S); // Training
        imgClass.trainOrTest = Scalar::all(1);
        imgClass.trainOrTest.resize(imgClass.features.size().height, Scalar::all(2)); // Testing

        rebalancedData.push_back(imgClass);
        total += imgClass.features.size().height;
        dataTraining.release();
        synthetic.release();
        dataRebalanced.release();
        dataTesting.release();
      }
    }
    else{
      rebalancedData.push_back(imbalancedData[eachClass]);
      total += imbalancedData[eachClass].features.size().height;
    }
  }

  numberOfImages.str("");
  numberOfImages << total;

  string name = csvSmote + "256c_100r_"+numberOfImages.str()+"i_smote.csv";
  arq.open(name.c_str(), ios::out);
  if (!arq.is_open()) {
    cout << "It is not possible to open the feature's file: " << name << endl;
    exit(-1);
  }
  cout << "---------------------------------------------------------------------------------------" << endl;
  cout << "Wrote on data file named " << name << endl;
  cout << "---------------------------------------------------------------------------------------" << endl;
  arq << total << '\t' << rebalancedData.size() << '\t' << rebalancedData[0].features.size().width << endl;
  for(std::vector<Classes>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
    for (h = 0; h < it->features.size().height; h++){
      arq << countImg << '\t' << it->classNumber << '\t' << it->trainOrTest.at<int>(h,0) << '\t';
      for (w = 0; w < it->features.size().width; w++){
        arq << it->features.at<float>(h, w) << " ";
      }
      arq << endl;
      countImg++;
    }
  }
  arq.close();
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
  int prev, count_diff, treino, num_images;
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
  count_diff = 0;
  for (i = 1; i < qtdClasses; i++) {
    NumberImgInClass(database, i, &num_images, &treino);
    count_diff = count_diff + abs(num_images - prev);
    prev = num_images;
  }
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
    samples = ceil(imgsInClass*fator);
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
      imagesTraining = vectorRand.size()*prob;
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
      fator -= id/(double)qtdClasses;
    }
  }
  return dir;
}

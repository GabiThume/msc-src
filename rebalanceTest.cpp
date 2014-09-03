/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalance.h"

/* Read the features of the file and save them in Mat data */
Mat readFeatures(const string& filename, Mat &classes, int &nClasses){

    int i, j;
    float features;
    Mat data;
    size_t n, d;
    ifstream myFile(filename.c_str());
    string line, infos, numImage, classe, numFeatures, numClasses, objetos;

    if(!myFile)
        throw exception();

    /* Read the first line, which contains the number of objects, classes and features */
    getline(myFile, infos);
    if (infos == "")
        return Mat();
    stringstream info(infos);
    getline(info, objetos, '\t');
    getline(info, numClasses, '\t');
    nClasses = atoi(numClasses.c_str());
    getline(info, numFeatures, '\t');

    n = atoi(objetos.c_str());
    d = atoi(numFeatures.c_str());

    /* Create a Mat named data with the file data provided */
    data.create(n, d, CV_32FC1);
    classes.create(n, 1, CV_32FC1);
    while (getline(myFile, line)) {
        stringstream vector_features(line);
        getline(vector_features, numImage, '\t');
        getline(vector_features, classe, '\t');
        i = atoi(numImage.c_str());
        j = 0;
        while(vector_features >> features) {
            data.at<float>(i, j) = (float)features;
            j++;
        }
        classes.at<float>(i, 0)=atoi(classe.c_str());
    }

    myFile.close();
    return data;
}

void classifica(string base, string features){

    Classifier c;
    Size size;
    int numClasses;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir;
    Mat data, classes;

    /* Feature extraction */
    descriptor(base.c_str(), features.c_str(), 4, 256, 1, 0, 0, 0, 0, 4);

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = readFeatures(name.c_str(), classes, numClasses);
            size = data.size();

            if (size.height != 0){
                c.bayes(data, classes, numClasses, prob, 20);
            }
        }
    }
}

int main(int argc, char const *argv[]){

    Rebalance r;

    if (argc != 3){
        cout << "\nUsage: ./smoteTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }

    cout << "\nRunning a rebalance test" << endl;

    /* Calculate before rebalance */
    classifica(string(argv[1]), string(argv[2]));

    /* */
    r.generate();

    /* Calculate after rebalance */
    classifica(string(argv[1]), string(argv[2]));

    return 0;
}

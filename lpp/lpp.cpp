#include <tapkee/tapkee.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/projection.hpp>
#include <algorithm>
#include <vector>
#include <iterator>
#include "tapkee/src/cli/util.hpp"
#include <numeric>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "classifier.h"

using namespace std;
using namespace tapkee;


/* Read the features and save them in Mat data */
Mat readFeatures(const string& filename, Mat &classes, int &nClasses){
    int i, j;
    float features;
    Mat data;
    size_t n, d;
    ifstream myFile(filename.c_str());
    string line, infos, numImage, classe, numFeatures, numClasses, objetos;
	//ofstream ofs("saida.csv");

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
    //cout << filename << " features " << d << " imagens " << n << endl; 
    while (getline(myFile, line)) {
        stringstream vector_features(line);
        stringstream temp;
        getline(vector_features, numImage, '\t');
        getline(vector_features, classe, '\t');
        i = atoi(numImage.c_str());
        j = 0;
        while(vector_features >> features) {
            data.at<float>(i, j) = (float)features;
            //temp << (float)features << ", ";
            j++;
        }
        //int len = temp.str().length();
        //temp.seekp(len-2);
        //temp << ";\n";
        //ofs << temp.str();
        classes.at<float>(i, 0)=atoi(classe.c_str());
    }

    myFile.close();
    return data;
}

int main(int argc, const char** argv)
{
    Classifier c;
    Mat data, classes;
    DIR *directory;
    struct dirent *arq;
    ifstream my_file;
    string name_arq, name_dir, name, outName;
    int nClasses, metodo, atributos, janela, i, j, dim;
    string delimiter = ",", dimStr;
    Size size;
    pair <int, int> minority(-1,-1);
    float prob = 0.5, r;
    
    if (argc != 3){
        cout << "\nUsage: ./lpp (1) (2)\n\n\t(1) Features Directory" << endl;
        cout << "\t(2) Dimensions\n" << endl;
        exit(-1);
    }
	name_dir = argv[1];
	dimStr = argv[2];
	dim = atoi(argv[2]);

    /* For each file on input directory, do the following operations */
    directory = opendir(name_dir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){
            name_arq = arq->d_name;
            name = name_dir + arq->d_name;
            outName = "../projections/LPP_"+dimStr+"_"+name_arq;
            my_file.open(name.c_str());
            if(my_file.good()){
                /* Read the feature vectors */
                data = readFeatures(name.c_str(), classes, nClasses);
                size = data.size();
                if (size.height != 0){
                    srand((unsigned int)time(NULL));
                    cout << endl << name.c_str();
                    ifstream ifs(name.c_str());
                    ofstream ofs(outName.c_str());

                    DenseMatrix input_data(size.height, size.width);
                    for (i = 0; i < size.height; i++){
                        for (j = 0; j < size.width; j++){
                            input_data(i,j) = data.at<float>(i,j);
                        }                
                    }

                    input_data.transposeInPlace();

                    // Some matriz is only eigenvalued when it is diagonizable
                    // One way to fixing it is to calculate Determinant and 
                    // add 0.0001 when Determinant is zero
                    if (abs(input_data.determinant()) == 0){
                        for (i = 0; i < size.height; i++){
                            for (j = 0; j < size.width; j++){
                                r = (float)rand()/(float)(RAND_MAX);
                                input_data(i,j) = input_data(i,j)+0.0001*r;
                            }
                        }
                    }
                    TapkeeOutput result = initialize()
                        .withParameters((method=LocalityPreservingProjections,
                                        num_neighbors=10, target_dimension=dim))
                        .embedUsing(input_data);
                    write_data(result.embedding, ofs, delimiter[0]);
                    ofs.close();

                    //Mat projections(size.height, size.width, CV_64F);
                    Mat projections(size.height, size.width, CV_32FC1);
                    for (i = 0; i < size.height; i++){
                        for (j = 0; j < size.width; j++){
                            projections.at<float>(i,j) = result.embedding(i,j);
                        }                
                    }
                    c.bayes(prob, 10, projections, classes, nClasses, minority, "");
                }
            }
            my_file.close();
       }
    }
	return 0;
}



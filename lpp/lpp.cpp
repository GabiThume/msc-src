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
    int nClasses, metodo, atributos, janela, i, j;
    string delimiter = ",";
    Size size;
    pair <int, int> minority(-1,-1);
    float prob = 0.5;
    
    if (argc < 2)
        exit(-1);

	name_dir = argv[1];
    //name_dir = "../../caracteristicas_corel/16/";
    /* For each file on input directory, do the following operations */
    directory = opendir(name_dir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){
            string out = "";
            name_arq = arq->d_name;
            name = name_dir + arq->d_name;
            outName = "../../result_lpp/LPP_"+name_arq+".csv";
            my_file.open(name.c_str());
            if(my_file.good()){
                /* Read the feature vectors */
                data = readFeatures(name.c_str(), classes, nClasses);
                size = data.size();
                if (size.height != 0){

                    cout << name.c_str() << endl;
                    ifstream ifs(name.c_str());
                    ofstream ofs(outName.c_str());

                    DenseMatrix input_data(size.height, size.width);
                    for (i = 0; i < size.height; i++){
                        for (j = 0; j < size.width; j++){
                            input_data(i,j) = data.at<float>(i,j);
                        }                
                    }

                    input_data.transposeInPlace();

                    // A matriz is only eigenvalued when it is diagonizable
                    // One way to fixing it is to calculate Determinant and 
                    // add 0.0001 when Determinant is zero
                    cout << " Determinant " << input_data.determinant() << endl;
                    if (input_data.determinant() == -0){
                        for (i = 0; i < size.height; i++){
                            for (j = 0; j < size.width; j++){
                                input_data(i,j) = input_data(i,j)+0.0001;
                            }
                        }
                    }
                    cout << "Data contains " << input_data.cols() << " feature vectors with dimension of " << input_data.rows() << endl;

                    TapkeeOutput result = initialize()
                        .withParameters((method=LocalityPreservingProjections,
                        num_neighbors=3))
                        .embedUsing(input_data);
                    //ofs << result.embedding.transpose();
                    //result.embedding.transposeInPlace();
                    write_data(result.embedding, ofs, delimiter[0]);
                    ofs.close();
                    
                    Mat projections(size.height, size.width, CV_32FC1);
                    for (i = 0; i < size.height; i++){
                        for (j = 0; j < size.width; j++){
                            projections.at<float>(i,j) = result.embedding(i,j);
                        }                
                    }//"../../result_lpp/analysis/LPP_"+name_arq
                    c.bayes(prob, 10, projections, classes, nClasses, minority, "");
                }
            }
            my_file.close();
       }
    }
	return 0;
}


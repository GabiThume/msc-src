/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
void imbalance(Mat original, Mat classes, int factor, int numClasses, Mat &imbalancedData, Mat &imbalancedClasses, int start, int end){

    int total = 0, pos = 0, i, num, samples;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    samples = ceil(samples/factor);

    imbalancedData.create(num, size.width, CV_32FC1);
    imbalancedClasses.create(num, 1, CV_32FC1);

    while (total < samples) {
        /* Generate a random position to select samples to crete the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = imbalancedData.row(total);
            original.row(pos).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            Mat tmp = imbalancedData.row(total);
            original.row(i).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(i,0);
            total++;
        }
    }

}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, smallerClass, amountSmote, start, end, neighbors;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir, baseDir, featuresDir;
    Mat data, minorityClass, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    pair <int, int> min(0,0);

    myFile.open("original.csv");
    myFile.close();
    myFile.open("smote_accuracy.csv");
    myFile.close();

    if (argc != 3){
        cout << "\nUsage: ./smoteTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }
    baseDir = string(argv[1]);
    featuresDir = string(argv[2]);

    /* Feature extraction from images */
    descriptor(baseDir.c_str(), featuresDir.c_str(), 4, 256, 1, 0, 0, 0, 0, 4, "");

    nameDir = string(featuresDir.c_str()) + "/";
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

                cout << "---------------------------------------------------------------" << endl;
                cout << endl << "Features vectors file: " << name.c_str() << endl << endl;
                cout << "---------------------------------------------------------------" << endl;
                cout << "Classification using original vectors" << endl;
                c.bayes(prob, 10, data, classes, numClasses, min, "original.csv");

                for (i = 2; i <= 10; i*=2){

                    cout << "---------------------------------------------------------------" << endl;
                    cout << endl <<  "Normal Bayes Classification for imbalanced classes" << endl << endl;
                    cout << "\tDivide the number of original samples by a factor of " << i << endl <<"\tto create a minority class:"<< endl;

                    Mat imbalancedClasses, imbalancedData;
                    /* Desbalancing Data */
                    c.findSmallerClass(imbalancedClasses, numClasses, smallerClass, start, end);
                    imbalance(data, classes, i, numClasses, imbalancedData, imbalancedClasses, start, end);
                    size = imbalancedData.size();
                    /* Classifying without rebalancing */                    
                    c.bayes(prob, 10, imbalancedData, imbalancedClasses, numClasses, min, "original.csv");
                    
                    /* Copy the feature data to minorityClass */
                    imbalancedData.rowRange(start,end).copyTo(minorityClass);
                    /* Amount of SMOTE % */
                    amountSmote = 100;
                    neighbors = 5;
                    /* Over-sampling the minority class */
                    synthetic = s.smote(minorityClass, amountSmote, neighbors);

                    /* Concatenate the minority class with the synthetic */
                    vconcat(minorityClass, synthetic, minorityOverSampled);
                    Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);

                    /* Select the majority classes */
                    imbalancedData.rowRange(end, size.height).copyTo(majority);
                    imbalancedClasses.rowRange(end, size.height).copyTo(majorityClasses);

                    /* Concatenate the feature samples and classes */
                    vconcat(minorityClasses, majorityClasses, newClasses);
                    vconcat(minorityOverSampled, majority, total);

                    cout << endl << "\tSMOTE: Synthetic Minority Over-sampling Technique" << endl;
                    cout << "\tAmount to SMOTE: " << amountSmote << "%" << endl;
                    c.bayes(prob, 10, total, newClasses, numClasses, min, "smote_accuracy.csv");

                    minorityOverSampled.release();
                    minorityClasses.release();
                    majority.release();
                    majorityClasses.release();
                    newClasses.release();
                    total.release();
                }
            }
            myFile.close();
           }
    }
    return 0;
}
/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
void imbalance(Mat original, Mat classes, int factor, int numClasses, Mat *imbalancedData, Mat *imbalancedClasses, int start, int end){

    int total = 0, pos = 0, i, num, samples;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    samples = ceil(samples/factor);

    (*imbalancedData).create(num, size.width, CV_32FC1);
    (*imbalancedClasses).create(num, 1, CV_32FC1);

    while (total < samples) {
        /* Generate a random position to select samples to crete the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = (*imbalancedData).row(total);
            original.row(pos).copyTo(tmp);
            (*imbalancedClasses).at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            Mat tmp = (*imbalancedData).row(total);
            original.row(i).copyTo(tmp);
            (*imbalancedClasses).at<float>(total, 0) = classes.at<float>(i,0);
            total++;
        }
    }

}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int i, smallerClass, amountSmote, neighbors;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir, baseDir, featuresDir;
    Mat minorityClass, classes, minorityOverSampled, majority, majorityClasses;
    Mat newClasses, total, synthetic, trainTest;
    pair <int, int> min(-1,-1);
    vector<Classes> data;

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
    vector<int> parameters;
    descriptor(baseDir.c_str(), featuresDir.c_str(), 4, 256, 1, 0, parameters, 0, 4, "");

    nameDir = string(featuresDir.c_str()) + "/";
    directory = opendir(nameDir.c_str());

    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = readFeatures(name);
            if (data.size() != 0){

                cout << "---------------------------------------------------------------" << endl;
                cout << endl << "Features vectors file: " << name.c_str() << endl << endl;
                cout << "---------------------------------------------------------------" << endl;
                cout << "Classification using original vectors" << endl;
                c.classify(prob, 10, data, "original.csv", 0);

                for (i = 2; i <= 10; i*=2){

                    cout << "---------------------------------------------------------------" << endl;
                    cout << endl <<  "Normal Bayes Classification for imbalanced classes" << endl << endl;
                    cout << "\tDivide the number of original samples by a factor of " << i << endl <<"\tto create a minority class:"<< endl;

                    Mat imbalancedClasses, imbalancedData;
                    // /* Desbalancing Data */
                    // c.findSmallerClass(data, &smallerClass);
                    // imbalance(data, i, &imbalancedData, smallerClass);
                    // size = imbalancedData.size();
                    // /* Classifying without rebalancing */
                    // c.classify(prob, 10, imbalancedData, "original.csv");

                    // /* Copy the feature data to minorityClass */
                    // imbalancedData.rowRange(start,end).copyTo(minorityClass);
                    // /* Amount of SMOTE % */
                    // amountSmote = minorityClass.size().height;
                    // neighbors = 5;
                    // /* Over-sampling the minority class */
                    // synthetic = s.smote(minorityClass, amountSmote, neighbors);

                    //  Concatenate the minority class with the synthetic
                    // vconcat(minorityClass, synthetic, minorityOverSampled);
                    //Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);

                    // /* Select the majority classes */
                    // imbalancedData.rowRange(end, size.height).copyTo(majority);
                    // imbalancedClasses.rowRange(end, size.height).copyTo(majorityClasses);

                    // /* Concatenate the feature samples and classes */
                    // vconcat(minorityClasses, majorityClasses, newClasses);
                    // vconcat(minorityOverSampled, majority, total);

                    // cout << endl << "\tSMOTE: Synthetic Minority Over-sampling Technique" << endl;
                    // cout << "\tAmount to SMOTE: " << amountSmote << "%" << endl;
                    // c.classify(prob, 10, total, newClasses, numClasses, min, trainTest, "smote_accuracy.csv");

                    minorityOverSampled.release();
                    //minorityClasses.release();
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

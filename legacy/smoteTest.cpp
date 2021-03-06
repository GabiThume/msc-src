/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "preprocessing/smote.h"

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
void imbalance(cv::Mat original, cv::Mat classes, int factor, int numClasses, cv::Mat *imbalancedData, cv::Mat *imbalancedClasses, int start, int end){

    int total = 0, pos = 0, i, num, samples;
    cv::Size size = original.size();
    std::vector<int> vectorRand;
    cv::Mat other, otherClasses;
    srand(time(0));

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    samples = ceil(samples/factor);

    (*imbalancedData).create(num, size.width, CV_64FC1);
    (*imbalancedClasses).create(num, 1, CV_64FC1);

    while (total < samples) {
        /* Generate a random position to select samples to crete the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            cv::Mat tmp = (*imbalancedData).row(total);
            original.row(pos).copyTo(tmp);
            (*imbalancedClasses).at<double>(total, 0) = classes.at<double>(start,0);
            total++;
       }
    }

    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            cv::Mat tmp = (*imbalancedData).row(total);
            original.row(i).copyTo(tmp);
            (*imbalancedClasses).at<double>(total, 0) = classes.at<double>(i,0);
            total++;
        }
    }

}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    cv::Size size;
    int i, smallerClass, amountSmote, neighbors;
    double prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    std::ifstream myFile;
    std::string nameFile, name, nameDir, baseDir, featuresDir;
    cv::Mat minorityClass, classes, minorityOverSampled, majority, majorityClasses;
    cv::Mat newClasses, total, synthetic, trainTest;
    pair <int, int> min(-1,-1);
    std::vector<ImageClass> data;

    myFile.open("original.csv");
    myFile.close();
    myFile.open("smote_accuracy.csv");
    myFile.close();

    if (argc != 3){
        std::cout << "\nUsage: ./smoteTest (1) (2)\n\n\t(1) Image Directory" << std::endl;
        std::cout << "\t(2) Features Directory\n" << std::endl;
        exit(-1);
    }
    baseDir = std::string(argv[1]);
    featuresDir = std::string(argv[2]);

    /* Feature extraction from images */
    std::vector<int> parameters;
    PerformFeatureExtraction(baseDir.c_str(), featuresDir.c_str(), 4, 64, 1, 1, parameters, 0, 4, "");

    nameDir = std::string(featuresDir.c_str()) + "/";
    directory = opendir(nameDir.c_str());

    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = ReadFeaturesFromFile(name);
            if (data.size() != 0){

                std::cout << "---------------------------------------------------------------" << std::endl;
                std::cout << std::endl << "Features vectors file: " << name.c_str() << std::endl << std::endl;
                std::cout << "---------------------------------------------------------------" << std::endl;
                std::cout << "Classification using original vectors" << std::endl;
                c.classify(prob, 10, data, "original.csv", 0);

                for (i = 2; i <= 10; i*=2){

                    std::cout << "---------------------------------------------------------------" << std::endl;
                    std::cout << std::endl <<  "Normal Bayes Classification for imbalanced classes" << std::endl << std::endl;
                    std::cout << "\tDivide the number of original samples by a factor of " << i << std::endl <<"\tto create a minority class:"<< std::endl;

                    cv::Mat imbalancedClasses, imbalancedData;
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
                    //cv::Mat minorityClasses(minorityOverSampled.size().height, 1, CV_64FC1, smallerClass+1);

                    // /* Select the majority classes */
                    // imbalancedData.rowRange(end, size.height).copyTo(majority);
                    // imbalancedClasses.rowRange(end, size.height).copyTo(majorityClasses);

                    // /* Concatenate the feature samples and classes */
                    // vconcat(minorityClasses, majorityClasses, newClasses);
                    // vconcat(minorityOverSampled, majority, total);

                    // std::cout << std::endl << "\tSMOTE: Synthetic Minority Over-sampling Technique" << std::endl;
                    // std::cout << "\tAmount to SMOTE: " << amountSmote << "%" << std::endl;
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

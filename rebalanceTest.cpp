/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"
#include "artificialGeneration.h"

void classifica(string dir, string features, string csvName, pair<int, int> min, int d, int m){

    Classifier c;
    Size size;
    int numClasses;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir, expected;
    Mat data, classes;
    Artificial a;
    stringstream numImages;
    numImages << min.second;
    const char* numImg = numImages.str().c_str();

    string methods[4] = {"Intensity", "Gleam", "Luminance", "MSB"};
    string descriptors[4] = {"BIC", "GCH", "CCV", "Haralick6"};

    a.generate(dir, 1);
    descriptor(dir.c_str(), features.c_str(), d, 256, 1, 0, 0, 0, 0, m, numImg);

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            expected = descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numImg+".txt";
            if(nameFile == expected){
                name = nameDir + arq->d_name;
                myFile.open(name.c_str());

                /* Read the feature vectors */
                data = readFeatures(name.c_str(), classes, numClasses);
                size = data.size();
                if (size.height != 0){
                    c.bayes(prob, 20, data, classes, numClasses, min, csvName.c_str());
                }
            }
        }
    }
}

void createDesbalancedFolders(int smallerClass, vector<int> vectorRand, int d, int m, string csvName){

    int x;
    string dir, str, nameFile, name, nameDir;
    stringstream numImages, smallClass;
    Size size;
    ifstream myFile;
    Mat data, classes;
    pair <int, int> min(smallerClass, vectorRand.size());

    numImages << vectorRand.size();
    smallClass << smallerClass;
    dir = "Desbalanced/"+numImages.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r Desbalanced/original/* "+dir+";";
    str += "rm "+dir+smallClass.str()+"/*;";
    system(str.c_str());
    /* Iterate */
    for(x = 0; x < (int)vectorRand.size(); x++){
        stringstream image;
        image << vectorRand[x];
        str = "cp Desbalanced/original/"+smallClass.str()+"/"+image.str()+".jpg ";
        str+= dir+smallClass.str()+"/";
        system(str.c_str());
    }
    str = "bash Desbalanced/rename.sh "+dir;
    system(str.c_str());
    classifica(dir, "features/artificial/", csvName, min, d, m);
}

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
void imbalance(Mat original, Mat classes, int factor, int numClasses, Mat &imbalancedData, Mat &imbalancedClasses, int d, int m, string csvName){

    int total = 0, pos = 0, i, smallerClass, start, end, samples, num;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));
    Classifier c;

    c.findSmallerClass(classes, numClasses, smallerClass, start, end);

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    samples = ceil(samples/factor);

    imbalancedData.create(num, size.width, CV_32FC1);
    imbalancedClasses.create(num, 1, CV_32FC1);

    while (total < samples) {
        /* Generate a random position to select samples to create the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = imbalancedData.row(total);
            original.row(pos).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    createDesbalancedFolders(smallerClass+1, vectorRand, d, m, csvName);

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
    int numClasses, i, smallerClass, amountSmote, start, end, neighbors, rep, m, d;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    ofstream csvFile;
    string nameFile, name, nameDir, baseDir, featuresDir, analysis, descriptorName, method;
    string csvOriginal, csvSmote, csvRebalance;
    Mat data, minorityClass, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    pair <int, int> min(-1,-1);

    if (argc != 3){
        cout << "\nUsage: ./rebalanceTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }
    baseDir = string(argv[1]);
    featuresDir = string(argv[2]);

    string methods[4] = {"Intensity", "Gleam", "Luminance", "MSB"};
    string descriptors[4] = {"BIC", "GCH", "CCV", "Haralick6"};

    //for (d = 4; d < 5; d++){
        //for (m = 4; m <= 4; m++){
            m = 4; d = 4;
            csvOriginal = "Desbalanced/analysis/original_accuracy_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = "Desbalanced/analysis/smote_accuracy_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = "Desbalanced/analysis/rebalance_accuracy_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            csvFile.open((csvOriginal+".csv").c_str(), ios::trunc);
            csvFile.close();
            csvFile.open((csvSmote+".csv").c_str(), ios::trunc);
            csvFile.close();
            csvFile.open((csvRebalance+".csv").c_str(), ios::trunc);
            csvFile.close();

            /* Feature extraction from images */
            descriptor(baseDir.c_str(), featuresDir.c_str(), d, 256, 1, 0, 0, 0, 0, m, "");

            nameDir = string(featuresDir.c_str()) + "/";
            directory = opendir(nameDir.c_str());

            if (directory != NULL){
                while ((arq = readdir(directory))){

                    nameFile = arq->d_name;

                    if (nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_.txt"){

                        name = nameDir + arq->d_name;
                        myFile.open(name.c_str());

                        /* Read the feature vectors */
                        data = readFeatures(name.c_str(), classes, numClasses);
                        size = data.size();
                        if (size.height != 0){
                            for (rep = 0; rep < 10; rep++){
                                cout << "---------------------------------------------------------------------------------------" << endl;
                                cout << "Classification using original vectors" << endl;
                                cout << "Features vectors file: " << name.c_str() << endl;
                                cout << "---------------------------------------------------------------------------------------" << endl;
                                c.bayes(prob, 20, data, classes, numClasses, min, csvOriginal.c_str());
                                c.bayes(prob, 20, data, classes, numClasses, min, csvSmote.c_str());
                                c.bayes(prob, 20, data, classes, numClasses, min, csvRebalance.c_str());
                                c.findSmallerClass(classes, numClasses, smallerClass, start, end);

                                for (i = 2; end-start > 2; i*=2){

                                    cout << "\n\n---------------------------------------------------------------------------------------" << endl;
                                    cout << "Divide the number of original samples by a factor of " << i << " to create a minority class:"<< endl;
                                    cout << "---------------------------------------------------------------------------------------" << endl;

                                    Mat imbalancedClasses, imbalancedData;
                                    /* Desbalancing Data */
                                    imbalance(data, classes, i, numClasses, imbalancedData, imbalancedClasses, d, m, csvRebalance);
                                    /* Classifying without rebalancing */
                                    c.bayes(prob, 20, imbalancedData, imbalancedClasses, numClasses, min, csvOriginal.c_str());

                                    c.findSmallerClass(imbalancedClasses, numClasses, smallerClass, start, end);
                                    /* Copy the feature data to minorityClass */
                                    imbalancedData.rowRange(start,end).copyTo(minorityClass);
                                    /* Amount of SMOTE % */
                                    amountSmote = 100;
                                    neighbors = 5;
                                    cout << endl << "SMOTE: Synthetic Minority Over-sampling Technique" << endl;
                                    cout << "Amount to SMOTE: " << amountSmote << "%" << endl;
                                    /* Over-sampling the minority class */
                                    synthetic = s.smote(minorityClass, amountSmote, neighbors);

                                    /* Concatenate the minority class with the synthetic */
                                    vconcat(minorityClass, synthetic, minorityOverSampled);
                                    Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);

                                    /* Select the majority classes */
                                    imbalancedData.rowRange(end, imbalancedData.size().height).copyTo(majority);
                                    imbalancedClasses.rowRange(end, imbalancedData.size().height).copyTo(majorityClasses);

                                    /* Concatenate the feature samples and classes */
                                    vconcat(minorityClasses, majorityClasses, newClasses);
                                    vconcat(minorityOverSampled, majority, total);
                                    pair <int, int> minSmote(smallerClass, end-start);
                                    c.bayes(prob, 20, total, newClasses, numClasses, minSmote, csvSmote.c_str());

                                    minorityOverSampled.release();
                                    minorityClasses.release();
                                    majority.release();
                                    majorityClasses.release();
                                    newClasses.release();
                                    total.release();
                                }
                            }
                        }
                        myFile.close();
                    }
                }
            }
        //}
    //}
    return 0;
}
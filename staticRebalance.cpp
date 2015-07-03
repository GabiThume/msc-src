/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC
 *
 **/

#include "smote.h"
#include "artificialGeneration.h"

string descriptors[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
string methods[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

string desc(string dir, string features, int d, int m, string id){

    int paramCCV[1] = {25};
    int paramACC[4] = {1, 3, 5, 7};
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramCCV, 1, 0, m, id.c_str());
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramACC, 4, 0, m, id.c_str());
    else
        return descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, 0, 0, 0, m, id.c_str());
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, smallerClass, amountSmote, start, end, neighbors, m, d;
    int initialMethod, endMethod, level;
    float prob = 0.5;
    ofstream csvFile;
    string nameFile, name, featuresDir, analysis, descriptorName, method;
    string csvOriginal, csvSmote, csvRebalance;
    Mat data, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced;

    if (argc != 3){
        cout << "\nUsage: ./staticRebalance (1) (2)\n\n\t(1) Configuration Directory" << endl;
        cout << "\t(2) ID of the generation\n" << endl;
        exit(-1);
    }
    string path = string(argv[1]);
    string id = string(argv[2]);

    string baseDirOriginal[3] = {"Balanced/", "25Desbalanced/", "12Desbalanced/"};
    featuresDir = path+"Features/";
    string baseDirID[3] = {"Balanced/", "25Rebalanced/", "12Rebalanced/"};
    int minorityNumber[3] = {50, 25, 12};

    /* Available Descriptors: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
        Quantization methods: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    for (d = 1; d <= 7; d++){
        initialMethod = 1;
        endMethod = 4;
        if (d < 6)
            endMethod = 1;
        if (d == 4){ // For Haralick use Intensity quantization only
            initialMethod = 1;
            endMethod = 1;
        }
        else if (d == 7){ // If it is HOG then use Intensity and Luminance quantization
            initialMethod = 1;
            endMethod = 2;
        }

        for (m = initialMethod; m <= endMethod; m++){
            csvOriginal = path+"Analysis/original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = path+"Analysis/smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = path+"Analysis/"+id+"_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            for (level = 0; level <= 2; level++){
                /* Feature extraction from images */
                string originalDescriptor = desc(path+baseDirOriginal[level], featuresDir, d, m, "original");
                string idDescriptor = desc(path+baseDirID[level], featuresDir, d, m, id);

                /* Read the original feature vectors */
                data = readFeatures(idDescriptor, &classes, &numClasses);
                size = data.size();
                if (size.height != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using " << id << endl;
                    cout << "Features vectors file: " << name.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;

                    pair <int, int> min(1, minorityNumber[level]);
                    c.bayes(prob, 10, data, classes, numClasses, min, csvRebalance.c_str());
                }
                data.release();

                /* Read the id feature vectors */
                data = readFeatures(originalDescriptor, &classes, &numClasses);
                size = data.size();
                if (size.height != 0){

                    pair <int, int> min(-1,-1);
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using original vectors" << endl;
                    cout << "Features vectors file: " << name.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.bayes(prob, 10, data, classes, numClasses, min, csvOriginal.c_str());

                    /* SMOTE */
                    // pair <int, int> min(1, minorityNumber[level]);
                    min.first = 1;
                    min.second = minorityNumber[level];
                    c.findSmallerClass(classes, numClasses, &smallerClass, &start, &end);
                    cout << "smaler class" << smallerClass << endl;
                    cout << "end " << end << " start " << start << endl;
                    /* Copy the feature data to minorityClass */
                    data.rowRange(start,min.second).copyTo(minorityTraining);
                    data.rowRange(min.second,end).copyTo(minorityTesting);

                    /* Amount of SMOTE % */
                    // amountSmote = ((imbalancedData.size().height-end-(end-start)) / (end-start))*100.0;
                    amountSmote = ((100-(end-start)) / (end-start))*100.0;
                    cout << "amountSmote: " << amountSmote << endl;
                    amountSmote = amountSmote == 0? 100 : amountSmote;
                    if (amountSmote > 0){
                        neighbors = amountSmote/100;
                        // neighbors = 5; //TODO: Fix amount to smote for more than one class
                        cout << endl << "SMOTE: Synthetic Minority Over-sampling Technique" << endl;
                        cout << "Amount to SMOTE: " << amountSmote << "%" << endl;
                        /* Over-sampling the minority class */
                        synthetic = s.smote(minorityTraining, amountSmote, neighbors);
                        cout << "minoritaria de treino " << minorityTraining.size().height << endl;
                        /* Concatenate the minority class with the synthetic */
                        vconcat(minorityTraining, synthetic, minorityRebalanced);
                        vconcat(minorityRebalanced, minorityTesting, minorityOverSampled);
                        //cout << "\nminority over " << minorityOverSampled.size().height << " minorityClass " << minorityTraining.size().height << " synthetic " << synthetic.size().height << endl;
                        Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);
                        // Select the majority classes 
                        data.rowRange(end, data.size().height).copyTo(majority);
                        classes.rowRange(end, data.size().height).copyTo(majorityClasses);

                        if (start != 0){ // Copy the initial majority
                            Mat majorityInitial, majorityClassesInitial;
                            data.rowRange(0, start).copyTo(majorityInitial);
                            classes.rowRange(0, start).copyTo(majorityClassesInitial);
                            vconcat(majority, majorityInitial, majority);
                            vconcat(majorityClasses, majorityClassesInitial, majorityClasses);
                        }
                        cout << "\nmajority " << majority.size().height << " end " << end << " start " << start << " Amount " << amountSmote << endl;
                        /* Concatenate the feature samples and classes */
                        vconcat(minorityClasses, majorityClasses, newClasses);
                        vconcat(minorityOverSampled, majority, total);
                        pair <int, int> minSmote(smallerClass+1, minorityRebalanced.size().height);
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using SMOTE" << endl;
                        cout << "Features vectors file: " << name.c_str() << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;

                        c.bayes(prob, 10, total, newClasses, numClasses, minSmote, csvSmote.c_str());

                        minorityOverSampled.release();
                        minorityClasses.release();
                        majority.release();
                        majorityClasses.release();
                        newClasses.release();
                        total.release();
                        synthetic.release();
                        minorityTraining.release();
                        minorityTesting.release();
                        minorityRebalanced.release();
                    }
                }
                data.release();
            }
        }
    }
    return 0;
}
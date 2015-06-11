/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"
#include "artificialGeneration.h"

string descriptors[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
string methods[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

// descriptor(databaseDir.c_str(), descMethodDir.c_str(), descMethod, colors, resize, normalization, params, numParameters, deleteNull, quantMethod, id.c_str());
void desc(string dir, string features, int d, int m, string id){

    int paramCCV[1] = {25};
    int paramACC[4] = {1, 3, 5, 7};
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramCCV, 1, 0, m, id.c_str());
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramACC, 4, 0, m, id.c_str());
    else
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, 0, 0, 0, m, id.c_str());
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, smallerClass, amountSmote, start, end, neighbors, m, d;
    int initialMethod, endMethod, level;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    ofstream csvFile;
    string nameFile, name, nameDir, featuresDir, analysis, descriptorName, method;
    string csvOriginal, csvSmote, csvRebalance;
    Mat data, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced;

    string baseDirOriginal[3] = {"PCASmote/Balanced/", "PCASmote/25Desbalanced/", "PCASmote/12Desbalanced/"};
    featuresDir = "PCASmote/Features/";
    string baseDirPCA[3] = {"PCASmote/Balanced/", "PCASmote/25Rebalanced/", "PCASmote/12Rebalanced/"};
    int minorityNumber[3] = {50, 25, 12};

    // string descriptors[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
    // string methods[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

    for (d = 6; d <= 6; d++){
        initialMethod = 1;
        endMethod = 4;
        if (d < 6)
            endMethod = 1;
        if (d == 4){ // For Haralick use Intensity quantization
            initialMethod = 1;
            endMethod = 1;
        }
        else if (d == 7){ // If it is HOG then use Intensity and Luminance quantization
            initialMethod = 1;
            endMethod = 2;
        }

        for (m = initialMethod; m <= endMethod; m++){
            csvOriginal = "PCASmote/Analysis/original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = "PCASmote/Analysis/smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = "PCASmote/Analysis/pca_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            for (level = 0; level <= 0; level++){
                /* Feature extraction from images */
                desc(baseDirOriginal[level], featuresDir, d, m, "original");
                desc(baseDirPCA[level], featuresDir, d, m, "pca");

                nameDir = string(featuresDir.c_str()) + "/";
                directory = opendir(nameDir.c_str());

                if (directory != NULL){
                    while ((arq = readdir(directory))){

                        nameFile = arq->d_name;
                        // TODO: mudar essas strings fixas
                        if (nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_.txt" ||
                            nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_4d_100r_.txt" ||
                            nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_PCA.txt" ||
                            nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_4d_100r_PCA.txt"){

                            cout << "ARQUIVO: " << nameFile << endl;

                            name = nameDir + arq->d_name;
                            myFile.open(name.c_str());

                            /* Read the original feature vectors */
                            data = readFeatures(&name, &classes, &numClasses);
                            // I am not longer able to share websites, maybe it is related? I already disable all the others plugins  
                            size = data.size();
                            if (size.height != 0){

                                cout << "---------------------------------------------------------------------------------------" << endl;
                                cout << "Classification using original vectors" << endl;
                                cout << "Features vectors file: " << name.c_str() << endl;
                                cout << "---------------------------------------------------------------------------------------" << endl;

                                /* Geradas com PCA */
                                if (nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_PCA.txt" ||
                                    nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_4d_100r_PCA.txt"){
                                    pair <int, int> min(1, minorityNumber[level]);

                                    c.bayes(prob, 10, data, classes, numClasses, min, csvRebalance.c_str());
                                }
                                else{
                                    pair <int, int> min(-1,-1);
                                    c.bayes(prob, 10, data, classes, numClasses, min, csvOriginal.c_str());


                                    /* SMOTE */
                                    // pair <int, int> min(1, minorityNumber[level]);
                                    min.first = 1;
                                    min.second = minorityNumber[level];
                                    c.findSmallerClass(classes, numClasses, &smallerClass, &start, &end);
                                    /* Copy the feature data to minorityClass */
                                    data.rowRange(start,min.second).copyTo(minorityTraining);
                                    data.rowRange(min.second,end).copyTo(minorityTesting);

                                    /* Amount of SMOTE % */
                                    // amountSmote = ((imbalancedData.size().height-end-(end-start)) / (end-start))*100.0;
                                    amountSmote = ((100-(end-start)) / (end-start))*100.0;
                                    cout << ">> end " << end << " start " << start << " amountSmote " << amountSmote << endl;
                                    // amountSmote = amountSmote == 0? 100 : amountSmote;
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
                                        // I am going to lose the internet connection now, I am sorry for the report @mo. Maybe I tried to upload to much stuff ;-)
                                        //cout << "\nmajority " << majority.size().height << " end " << end << " start " << start << " Amount " << amountSmote << endl;
                                        /* Concatenate the feature samples and classes */
                                        vconcat(minorityClasses, majorityClasses, newClasses);
                                        vconcat(minorityOverSampled, majority, total);
                                        pair <int, int> minSmote(smallerClass+1, minorityRebalanced.size().height);
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
                            }
                            data.release();
                        }
                        myFile.close();
                    }
                }
            }
        }
    }
    return 0;
}
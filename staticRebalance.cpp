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

string intToString(int number){
    stringstream s;
    s << number;
    string r = s.str();
    return r;
}

vector<Classes> performSmote(vector<Classes> imbalancedData, int *total){

    int majority = -1, majorityClass = -1, eachClass, amountSmote;
    int numTraining = 0, numTesting = 0, i, x, neighbors;
    vector<int> trainingNumber(imbalancedData.size(), 0);
    std::vector<Classes>::iterator it;
    vector<Classes> rebalancedData;
    Mat synthetic;
    SMOTE s;

    for (it = imbalancedData.begin(); it != imbalancedData.end(); ++it){
        if (it->fixedTrainOrTest){
            for(i = 0; i < it->trainOrTest.size().height; ++i){
                if (it->trainOrTest.at<float>(i,0) == 1)
                    trainingNumber[it->classNumber]++;
            }
        }
        else{
            trainingNumber[it->classNumber] = it->trainOrTest.size().height/2;
        }
        if (trainingNumber[it->classNumber] > majority){
            majorityClass = it->classNumber;
            majority = trainingNumber[it->classNumber];
        }
    }

    (*total) = 0;
    for (eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass){

        Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_32FC1);
        Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_32FC1);

        numTraining = 0;
        numTesting = 0;

        cout << "In class " << eachClass << " original images: " << trainingNumber[eachClass] << endl;
        /* Find out how many samples are needed to rebalance */
        amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];

        if (amountSmote > 0){
            //neighbors = 5;
            neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
            //cout << " neighbors " << neighbors << endl;

            for (x = 0; x < imbalancedData[eachClass].trainOrTest.size().height; ++x){
                if (imbalancedData[eachClass].trainOrTest.at<float>(x,0) == 1){
                    dataTraining.resize(numTraining+1);
                    Mat tmp = dataTraining.row(numTraining);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTraining++;
                }
                if (imbalancedData[eachClass].trainOrTest.at<float>(x,0) == 2){
                    dataTesting.resize(numTesting+1);
                    Mat tmp = dataTesting.row(numTesting);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTesting++;
                }
            }

            synthetic = s.smote(dataTraining, amountSmote, neighbors);

            cout << " new synthetic samples: " << amountSmote << endl;

            /* Concatenate original with synthetic data*/
            Classes imgClass;
            Mat dataRebalanced;
            vconcat(dataTraining, synthetic, dataRebalanced);
            vconcat(dataRebalanced, dataTesting, imgClass.features);
            imgClass.classNumber = eachClass;
            imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32FC1); // Training
            imgClass.trainOrTest = Scalar(1);
            imgClass.trainOrTest.resize(imgClass.features.size().height, 2); // Testing

            rebalancedData.push_back(imgClass);
            (*total) += imgClass.features.size().height;
            dataTraining.release();
            synthetic.release();
            dataRebalanced.release();
            dataTesting.release();
        }
        else{
            rebalancedData.push_back(imbalancedData[eachClass]);
            (*total) += imbalancedData[eachClass].features.size().height;
        }
    }

    // FILE *arq = fopen("smote.data", "w+");
    // int w, z;
    // fprintf(arq,"%s\n", "DY");
    // fprintf(arq,"%d\n", newClasses.size().height);
    // fprintf(arq,"%d\n", total.size().width);
    // for(z = 0; z < total.size().width-1; z++) {
    //     fprintf(arq,"%s%d;", "attr",z);
    // }
    // fprintf(arq,"%s%d\n", "attr",z);
    // for (w = 0; w < newClasses.size().height; w++) {
    //     fprintf(arq,"%d%s;", w,".jpg");
    //     for(z = 0; z < total.size().width; z++) {
    //         fprintf(arq,"%.5f;", total.at<float>(w, z));
    //     }
    //     fprintf(arq,"%1.1f\n", newClasses.at<float>(w,0));
    // }
    return rebalancedData;
}

int main(int argc, char const *argv[]){

    Classifier c;
    Size size;
    int m, d, h, w, totalRebalanced;
    int initialMethod, endMethod, level;
    float prob = 0.5;
    ofstream csvFile;
    string nameFile, name, featuresDir, analysis, descriptorName, method;
    string csvOriginal, csvSmote, csvRebalance;
    vector<Classes> originalData, imbalancedData, artificialData;

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

        for (m = initialMethod; m <= endMethod; m++){
            csvOriginal = path+"Analysis/original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = path+"Analysis/smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = path+"Analysis/"+id+"_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            for (level = 0; level <= 2; level++){
                /* Feature extraction from images */
                string originalDescriptor = desc(path+baseDirOriginal[level], featuresDir, d, m, "original");
                // string idDescriptor = desc(path+baseDirID[level], featuresDir, d, m, id);
                originalData = readFeatures(originalDescriptor);
                if (originalData.size() != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using original vectors" << endl;
                    cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.classify(prob, 1, originalData, csvOriginal.c_str(), minorityNumber[level]);
                    originalData.clear();
                }

                string dirRebalanced = path+baseDirID[level];
                string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
                artificialData = readFeatures(fileDescriptor);
                if (artificialData.size() != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using rebalanced data" << endl;
                    cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.classify(prob, 1, artificialData, csvRebalance.c_str(), minorityNumber[level]);
                    artificialData.clear();
                }

                /* Generate Synthetic SMOTE samples */
                originalData = readFeatures(originalDescriptor);
                vector<Classes> rebalancedData = performSmote(originalData, &totalRebalanced);
                if (rebalancedData.size() != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using SMOTE" << endl;
                    cout << "Features vectors file: " << name.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.classify(prob, 1, rebalancedData, csvSmote.c_str(), minorityNumber[level]);

                    stringstream numberOfImages;
                    numberOfImages.str("");
                    numberOfImages << totalRebalanced;
                    string name = featuresDir+descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";

                    FILE *arq = fopen(name.c_str(), "w+");
                    fprintf(arq,"%d %d\t%d\n", totalRebalanced, rebalancedData.size(), (int) rebalancedData[0].features.size().width);
                    int imgNumber = 0;
                    for(std::vector<Classes>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
                        for (h = 0; h < it->features.size().height; h++, imgNumber++){
                            fprintf(arq,"%d\t%d\t%d\t", imgNumber, it->classNumber, (int) it->trainOrTest.at<float>(h,0));
                            for (w = 0; w < it->features.size().width; w++){
                                fprintf(arq,"%.5f ", it->features.at<float>(h, w));
                            }
                            fprintf(arq,"\n");
                        }
                    }
                    rebalancedData.clear();
                }
            }
        }
    }
    return 0;
}

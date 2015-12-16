/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC
 *
 **/

#include "rebalanceTest.h"

std::string desc(std::string dir, std::string features, int d, int m, std::string id){

    std::vector<int> paramCCV = {25};
    std::vector<int> paramACC = {1, 3, 5, 7};
    std::vector<int> parameters;
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramCCV, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramACC, 0, m, id);
    else
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, parameters, 0, m, id);
}

std::string intToString(int number){
    std::stringstream s;
    s << number;
    std::string r = s.str();
    return r;
}

std::vector<ImageClass> performSmote(std::vector<ImageClass> imbalancedData, int *total){

    int majority = -1, majorityClass = -1, eachClass, amountSmote;
    int numTraining = 0, numTesting = 0, i, x, neighbors;
    std::vector<int> trainingNumber(imbalancedData.size(), 0);
    std::vector<ImageClass>::iterator it;
    std::vector<ImageClass> rebalancedData;
    cv::Mat synthetic;
    SMOTE s;

    for (it = imbalancedData.begin(); it != imbalancedData.end(); ++it){
        if (it->fixedTrainOrTest){
            for(i = 0; i < it->trainOrTest.size().height; ++i){
                if (it->trainOrTest.at<double>(i,0) == 1)
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

        cv::Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_64FC1);
        cv::Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_64FC1);

        numTraining = 0;
        numTesting = 0;

        std::cout << "In class " << eachClass << " original images: " << trainingNumber[eachClass] << std::endl;
        /* Find out how many samples are needed to rebalance */
        amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];

        if (amountSmote > 0){
            //neighbors = 5;
            neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
            //std::cout << " neighbors " << neighbors << std::endl;

            for (x = 0; x < imbalancedData[eachClass].trainOrTest.size().height; ++x){
                if (imbalancedData[eachClass].trainOrTest.at<double>(x,0) == 1){
                    dataTraining.resize(numTraining+1);
                    cv::Mat tmp = dataTraining.row(numTraining);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTraining++;
                }
                if (imbalancedData[eachClass].trainOrTest.at<double>(x,0) == 2){
                    dataTesting.resize(numTesting+1);
                    cv::Mat tmp = dataTesting.row(numTesting);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTesting++;
                }
            }

            synthetic = s.smote(dataTraining, amountSmote, neighbors);

            std::cout << " new synthetic samples: " << amountSmote << std::endl;

            /* Concatenate original with synthetic data*/
            Classes imgClass;
            cv::Mat dataRebalanced;
            vconcat(dataTraining, synthetic, dataRebalanced);
            vconcat(dataRebalanced, dataTesting, imgClass.features);
            imgClass.classNumber = eachClass;
            imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_64FC1); // Training
            imgClass.trainOrTest = cv::Scalar(1);
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
    //         fprintf(arq,"%.5f;", total.at<double>(w, z));
    //     }
    //     fprintf(arq,"%1.1f\n", newClasses.at<double>(w,0));
    // }
    return rebalancedData;
}

int main(int argc, char const *argv[]){

    Classifier c;
    cv::Size size;
    int m, d, h, w, totalRebalanced, countImg, initialMethod, endMethod, level;
    double prob = 0.5;
    std::ofstream arq;
    std::string nameFile, name, featuresDir, analysis, descriptorName, method;
    std::string csvOriginal, csvSmote, csvRebalance;
    std::vector<ImageClass> originalData, imbalancedData, artificialData;

    if (argc != 3){
        std::cout << "\nUsage: ./staticRebalance (1) (2)\n\n\t(1) Configuration Directory" << std::endl;
        std::cout << "\t(2) ID of the generation\n" << std::endl;
        exit(-1);
    }
    std::string path = std::string(argv[1]);
    std::string id = std::string(argv[2]);
    std::string baseDirOriginal[3] = {"Balanced/", "25Desbalanced/", "12Desbalanced/"};
    featuresDir = path+"Features/";
    std::string baseDirID[3] = {"Balanced/", "25Rebalanced/", "12Rebalanced/"};
    int minorityNumber[3] = {50, 25, 12};

    /* Available Descriptors: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
        Quantization methods: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    for (d = 1; d <= 7; d++){
        initialMethod = 1;
        endMethod = 4;

        for (m = initialMethod; m <= endMethod; m++){
            csvOriginal = path+"Analysis/original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
            csvSmote = path+"Analysis/smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
            csvRebalance = path+"Analysis/"+id+"_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
            for (level = 0; level <= 2; level++){
                /* Feature extraction from images */
                std::string originalDescriptor = desc(path+baseDirOriginal[level], featuresDir, d, m, "original");
                // std::string idDescriptor = desc(path+baseDirID[level], featuresDir, d, m, id);
                originalData = ReadFeaturesFromFile(originalDescriptor);
                if (originalData.size() != 0){
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    std::cout << "Classification using original vectors" << std::endl;
                    std::cout << "Features vectors file: " << originalDescriptor.c_str() << std::endl;
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    c.classify(prob, 1, originalData, csvOriginal.c_str(), minorityNumber[level]);
                    originalData.clear();
                }

                std::string dirRebalanced = path+baseDirID[level];
                std::string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
                artificialData = ReadFeaturesFromFile(fileDescriptor);
                if (artificialData.size() != 0){
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    std::cout << "Classification using rebalanced data" << std::endl;
                    std::cout << "Features vectors file: " << fileDescriptor.c_str() << std::endl;
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    c.classify(prob, 1, artificialData, csvRebalance.c_str(), minorityNumber[level]);
                    artificialData.clear();
                }

                /* Generate Synthetic SMOTE samples */
                originalData = ReadFeaturesFromFile(originalDescriptor);
                std::vector<ImageClass> rebalancedData = performSmote(originalData, &totalRebalanced);
                if (rebalancedData.size() != 0){
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    std::cout << "Classification using SMOTE" << std::endl;
                    std::cout << "Features vectors file: " << name.c_str() << std::endl;
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    c.classify(prob, 1, rebalancedData, csvSmote.c_str(), minorityNumber[level]);

                    std::stringstream numberOfImages;
                    numberOfImages.str("");
                    numberOfImages << totalRebalanced;
                    std::string name = featuresDir+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";

                    arq.open(name.c_str(), std::ios::out);
                    if (!arq.is_open()) {
                      std::cout << "It is not possible to open the feature's file: " << name << std::endl;
                      exit(-1);
                    }
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    std::cout << "Wrote on data file named " << name << std::endl;
                    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                    arq << totalRebalanced << '\t' << rebalancedData.size() << '\t' << rebalancedData[0].features.cols << std::endl;
                    for(std::vector<ImageClass>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
                        for (h = 0; h < it->features.size().height; h++){
                            arq << countImg << '\t' << it->classNumber << '\t' << it->trainOrTest.at<int>(h,0) << '\t';
                            for (w = 0; w < it->features.size().width; w++){
                              arq << it->features.at<float>(h, w) << " ";
                            }
                            arq << std::endl;
                            countImg++;
                        }
                    }
                    arq.close();
                    rebalancedData.clear();
                }
            }
        }
    }
    return 0;
}

/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalanceTest.h"

string descriptors[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
string methods[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

// descriptor(databaseDir.c_str(), descMethodDir.c_str(), descMethod, colors, resize, normalization, params, numParameters, deleteNull, quantMethod, id.c_str());
string desc(string dir, string features, int d, int m, string id){

    int paramCCV[1] = {25};
    int paramACC[4] = {1, 3, 5, 7};

    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return descriptor(dir, features, d, 256, 1, 1, paramCCV, 1, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return descriptor(dir, features, d, 256, 1, 1, paramACC, 4, 0, m, id);
    else
        return descriptor(dir, features, d, 256, 1, 1, 0, 0, 0, m, id);
}

/* Generate a imbalanced class */
string imbalance(vector<Classes> imageClasses, string baseDir, string newDir, double prob){

    int pos = 0, samples, imagesTraining;
    string dir;
    vector<int> vectorRand;
    srand(time(0));
    int x;
    string str, nameFile, name, nameDir;
    stringstream numImages, classNumber, image;
    Size size;
    ifstream myFile;
    Mat data, classes;
    double fator = 1.0;

    int allImages = 0;
    for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {
        allImages += it->features.size().height;
    }

    dir = newDir+"/steps/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r "+baseDir+"* "+dir+";";
    //cout << " Executa " << str.c_str() << endl;
    system(str.c_str());

    //fator = 1;
    for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {
        // samples = it->features.size().height - it->features.size().height/fator;
        samples = it->features.size().height*fator;
        //cout << " it->features.size().height " << it->features.size().height << " fator " << fator << " samples " << samples << endl;
        /* Generate a random position to select samples to create the minority class */

        if (samples > 0){

            numImages.str("");
            numImages << vectorRand.size();
            classNumber.str("");
            classNumber << it->classNumber;

            //str = "rm "+dir+classNumber.str()+"/*;";
            str = "rm -f -r "+dir+classNumber.str()+"/*;";
            str += "mkdir -p "+dir+classNumber.str()+"/treino/;";
            str += "mkdir -p "+dir+classNumber.str()+"/teste/;";

            //str = "mkdir -p "+dir+classNumber.str()+"/teste/ "+dir+classNumber.str()+"/treino/;";
            //cout << " Executa " << str.c_str() << endl;
            system(str.c_str());

            while ((int)vectorRand.size() < samples) {
                pos = rand() % it->features.size().height;
                if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                    vectorRand.push_back(pos);
               }
            }

            // dir = newDir+numImages.str()+"/";
            // str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
            // str += "cp -r "+baseDir+"* "+dir+";";
            // str += "rm "+dir+classNumber.str()+"/*;";
            // str += "mkdir -p "+dir+classNumber.str()+"/teste/ "+dir+classNumber.str()+"/treino/;";
            // cout << " Executa " << str.c_str() << endl;
            //system(str.c_str());

            /* Copy some of the originals in treino vs teste */
            imagesTraining = vectorRand.size()*prob;
            for(x = 0; x < imagesTraining; x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+baseDir+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/treino/; ";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            for(x = imagesTraining; x < (int)vectorRand.size(); x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+baseDir+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/teste/;";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            /* Copy the rest of originals in teste */
            
            str = "bash "+newDir+"rename.sh "+dir+classNumber.str()+"/";

            //cout << " Executa " << str.c_str() << endl;
            system(str.c_str());

            vectorRand.clear();
        }
        fator -= 0.1;
        //fator *= 2;
    }
    return dir;
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, amountSmote, neighbors, rep, m, d, operation;
    int initialMethod, endMethod, x, h, w;
    float prob = 0.5;
    ofstream csvFile;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
    Mat classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced, trainTest;
    Mat trainTestOriginal;
    Artificial a;
    stringstream numImages;
    vector<Classes> data, imbalancedData, artificialData;
    std::vector<Classes>::iterator it;

    if (argc != 6){
        cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
        cout << "\t(0) Directory to place tests\n" << endl;
        cout << "\t(1) Image Directory\n" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        cout << "\t(3) Analysis Directory\n" << endl;
        cout << "\t(4) Artificial Generation Operation - Use 0 for ALL\n" << endl;
        cout << "\t./rebalanceTest Desbalanced/ Desbalanced/original/ Desbalanced/features/ Desbalanced/analysis/ 0\n" << endl;
        exit(-1);
    }
    newDir = string(argv[1]);
    baseDir = string(argv[2]);
    featuresDir = string(argv[3]);
    analysisDir = string(argv[4]);
    operation = atoi(argv[5]);
    string op = argv[5];

    /* 
        Available 
            Descriptors: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
            Quantization methods: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    for (d = 1; d <= 7; d++){
        initialMethod = 1;
        endMethod = 1;
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
            csvOriginal = analysisDir+op+"-original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = analysisDir+op+"-smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = analysisDir+op+"-artificial_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            /* Feature extraction from images */
            string fileDescriptor = desc(baseDir, featuresDir, d, m, "original");
            /* Read the feature vectors */
            data = readFeatures(fileDescriptor);
            numClasses = data.size();
            if (numClasses != 0){
                // for (rep = 0; rep < 1; rep++){
                    // cout << "---------------------------------------------------------------------------------------" << endl;
                    // cout << "Classification using original vectors" << endl;
                    // cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                    // cout << "---------------------------------------------------------------------------------------" << endl;
                    // c.classify(prob, 10, data, csvOriginal.c_str(), 1);
                    // c.classify(prob, 10, data, csvSmote.c_str(), 1);
                    // c.classify(prob, 10, data, csvRebalance.c_str(), 1);

                    cout << "\n\n------------------------------------------------------------------------------------" << endl;
                    cout << "Divide the number of original samples to create a minority class:" << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;

                    /* Desbalancing Data */
                    string dirImbalanced = imbalance(data, baseDir, newDir, 0.5);
                    string originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "original");

                    imbalancedData = readFeatures(originalDescriptor);
                    numClasses = data.size();
                    if (numClasses != 0){
                        /* Classifying without rebalancing */
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using desbalanced data" << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        c.classify(prob, 10, imbalancedData, csvDesbalanced.c_str(), 2);
                    }


                    a.generate(dirImbalanced, operation);
                    string fileDescriptor = desc(dirImbalanced, featuresDir, d, m, "artificial");
                    artificialData = readFeatures(fileDescriptor);

                    numClasses = data.size();
                    if (numClasses != 0){
                        /* Classify with rebalanced data */
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using rebalanced data" << endl;
                        cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        c.classify(prob, 10, artificialData, csvRebalance.c_str(), 2);
                    }

                    int majority = -1, majorityClass = -1, eachClass;
                    vector<int> trainingNumber(imbalancedData.size(), 0);
                    for(it = imbalancedData.begin(); it != imbalancedData.end(); ++it) {
                        if (it->fixedTrainOrTest){
                            for(i = 0; i < it->trainOrTest.size().height; i++){
                                if (it->trainOrTest.at<float>(i,0) == 1)
                                    trainingNumber[it->classNumber]++;
                            }
                        }
                        else{
                            trainingNumber[it->classNumber] = it->trainOrTest.size().height/2; //TODO fixed
                        }
                        if (trainingNumber[it->classNumber] > majority){
                            majorityClass = it->classNumber;
                            majority = trainingNumber[it->classNumber];
                        }
                    }

                    vector<Classes> rebalancedData;
                    int total = 0;
                    for(eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass) {

                        Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_32FC1);
                        Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_32FC1);

                        int numTraining = 0, numTesting = 0;
                        
                        cout << "In class " << eachClass << " original images: " << trainingNumber[eachClass];
                        /* Find out how many samples are needed to rebalance */
                        amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];
 
                        if (amountSmote > 0){
                            //neighbors = 5; 
                            neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
                            //cout << " neighbors " << neighbors << endl;

                            for (x = 0; x < imbalancedData[eachClass].trainOrTest.size().height; x++){
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
                            total += imgClass.features.size().height;
                            dataTraining.release();
                            synthetic.release();
                            dataRebalanced.release();
                            dataTesting.release();
                        }
                        else{
                            rebalancedData.push_back(imbalancedData[eachClass]);
                            total += imbalancedData[eachClass].features.size().height;
                        }
                    
                    }

                    stringstream numberOfImages;
                    numberOfImages.str("");
                    numberOfImages << total;
                    string name = featuresDir+descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";

                    FILE *arq = fopen(name.c_str(), "w+");
                    fprintf(arq,"%d %d\t%d\n", total, rebalancedData.size(), (int) rebalancedData[0].features.size().width);
                    for(it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
                        for (h = 0; h < it->features.size().height; h++){
                            fprintf(arq,"%d\t%d\t%d\t", w, it->classNumber, (int) it->trainOrTest.at<float>(h,0));  
                            for (w = 0; w < it->features.size().width; w++){
                                fprintf(arq,"%.5f ", it->features.at<float>(h, w));
                            }
                            fprintf(arq,"\n");  
                        }
                    }

                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using SMOTE" << endl;
                    cout << "Features vectors file: " << name.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.classify(prob, 10, rebalancedData, csvSmote.c_str(), 2);

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
            }
            data.clear();
        }
    }
    return 0;
}
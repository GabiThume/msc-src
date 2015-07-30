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
string imbalance(vector<Classes> imageClasses, string baseDir, string newDir){

    int pos = 0, samples, fator;
    string dir;
    vector<int> vectorRand;
    srand(time(0));
    int x;
    string str, nameFile, name, nameDir;
    stringstream numImages, classNumber, image;
    Size size;
    ifstream myFile;
    Mat data, classes;

    int allImages = 0;
    for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {
        allImages += it->features.size().height;
    }

    fator = allImages/imageClasses.size();

    for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {
        samples = it->features.size().height - fator;

        /* Generate a random position to select samples to create the minority class */
        while ((int)vectorRand.size() < samples) {
            pos = rand() % it->features.size().height;
            if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                vectorRand.push_back(pos);
           }
        }

        numImages.str("");
        numImages << vectorRand.size();
        classNumber.str("");
        classNumber << it->classNumber;

        dir = newDir+numImages.str()+"/";
        str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";  
        str += "cp -r "+baseDir+"* "+dir+";";
        str += "rm "+dir+classNumber.str()+"/*;";
        str += "mkdir -p "+dir+classNumber.str()+"/teste/ "+dir+classNumber.str()+"/treino/;";

        cout << " Executa " << str.c_str() << endl;
        //system(str.c_str());

        /* Copy some of the originals in treino vs teste */
        for(x = 0; x < (int)vectorRand.size(); x++){
            image.str("");
            image << vectorRand[x];
            str = "cp "+baseDir+classNumber.str()+"/"+image.str()+".jpg ";
            str+= dir+classNumber.str()+"/treino/; ";
            x++;
            image.str("");
            image << vectorRand[x];
            str += "cp "+baseDir+classNumber.str()+"/"+image.str()+".jpg ";
            str+= dir+classNumber.str()+"/teste/;";
            system(str.c_str());
        }
        /* Copy the rest of originals in teste */
        
        str = "bash "+newDir+"rename.sh "+dir+classNumber.str()+"/";

        cout << " Executa " << str.c_str() << endl;
        //system(str.c_str());

        vectorRand.clear();
    }
    return dir;
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, amountSmote, neighbors, rep, m, d, operation;
    int initialMethod, endMethod, x;
    float prob = 0.5;
    ofstream csvFile;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir;
    Mat classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced, trainTest;
    Mat trainTestOriginal;
    Artificial a;
    stringstream numImages;
    vector<Classes> data, imbalancedData, artificialData;

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

    /* Available Descriptors: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
        Quantization methods: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    for (d = 1; d <= 1; d++){
        initialMethod = 1;
        endMethod = 1;
        // if (d < 6)
        //     endMethod = 1;
        // if (d == 4){ // For Haralick use Intensity quantization
        //     initialMethod = 1;
        //     endMethod = 1;
        // }
        // else if (d == 7){ // If it is HOG then use Intensity and Luminance quantization
        //     initialMethod = 1;
        //     endMethod = 2;
        // }

        for (m = initialMethod; m <= endMethod; m++){
            csvOriginal = analysisDir+op+"-original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = analysisDir+op+"-smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = analysisDir+op+"-artificial_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            /* Feature extraction from images */
            string fileDescriptor = desc(baseDir, featuresDir, d, m, "original");
            /* Read the feature vectors */
            data = readFeatures(fileDescriptor);
            numClasses = data.size();
            if (numClasses != 0){
                // for (rep = 0; rep < 1; rep++){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using original vectors" << endl;
                    cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.classify(prob, 10, data, csvOriginal.c_str());
                    c.classify(prob, 10, data, csvSmote.c_str());
                    c.classify(prob, 10, data, csvRebalance.c_str());

                    cout << "\n\n------------------------------------------------------------------------------------" << endl;
                    cout << "Divide the number of original samples to create a minority class:"<< endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;

                    // /* Desbalancing Data */
                    // string dirImbalanced = imbalance(data, baseDir, newDir);
                    // string originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "original");
                    // Mat trainTestImbalanced;
                    // imbalancedData = readFeatures(originalDescriptor);
                    // numClasses = data.size();
                    // if (numClasses != 0){
                    //     /* Classifying without rebalancing */
                    //     cout << "---------------------------------------------------------------------------------------" << endl;
                    //     cout << "Classification using desbalanced data" << endl;
                    //     cout << "---------------------------------------------------------------------------------------" << endl;
                    //     c.classify(prob, 10, imbalancedData, csvOriginal.c_str());
                    // }

                    // // a.generate(dirImbalanced, operation);
                    // // string fileDescriptor = desc(dirImbalanced, featuresDir, d, m, "artificial");
                    // // artificialData = readFeatures(fileDescriptor);
                    // // numClasses = data.size();
                    // // if (numClasses != 0){
                    // //     /* Classify with rebalanced data */
                    // //     cout << "---------------------------------------------------------------------------------------" << endl;
                    // //     cout << "Classification using rebalanced data" << endl;
                    // //     cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                    // //     cout << "---------------------------------------------------------------------------------------" << endl;
                    // //     c.classify(prob, 10, artificialData, csvRebalance.c_str());
                    // // }

                    // // vector<Classes> rebalancedData;
                    // // int majority = -1, majorityClass = -1, eachClass;
                    // // vector<int> totalImage;
                    // // for(std::vector<Classes>::iterator it = artificialData.begin(); it != artificialData.end(); ++it) {
                    // //     totalImage.push_back(it->features.size().height);
                    // //     if (it->features.size().height > majority){
                    // //         majorityClass = totalImage.size()-1;
                    // //         majority = it->features.size().height;
                    // //     }
                    // // }

                    // // for(eachClass = 0; eachClass < (int) totalImage.size(); ++eachClass) {
                    // //     Mat dataTraining, dataTesting, overSampled, dataRebalanced;
                    // //     int numTraining = 0, numTesting = 0;

                    // //     /* Find out how many samples are needed to rebalance */
                    // //     amountSmote = totalImage[majorityClass]/2 - totalImage[eachClass];
                    // //     if (amountSmote > 0){
                    // //         neighbors = 5;

                    // //         if (artificialData[eachClass].fixedTrainOrTest) {
                    // //             for (x = 0; x < artificialData[eachClass].trainOrTest.size().height; x++){
                    // //                 if (artificialData[eachClass].trainOrTest.at<float>(x,0) == 1){
                    // //                     Mat tmp = dataTraining.row(numTraining);
                    // //                     artificialData[eachClass].features.row(x).copyTo(tmp);
                    // //                     numTraining++;
                    // //                 }
                    // //                 if (artificialData[eachClass].trainOrTest.at<float>(x,0) == 2){
                    // //                     Mat tmp = dataTesting.row(numTesting);
                    // //                     artificialData[eachClass].features.row(x).copyTo(tmp);
                    // //                     numTesting++;
                    // //                 }
                    // //             }
                    // //         }

                    // //         synthetic = s.smote(dataTraining, amountSmote, neighbors);
    
                    // //         /* Concatenate original with synthetic data*/
                    // //         Classes imgClass;
                    // //         vconcat(dataTraining, synthetic, dataRebalanced);
                    // //         vconcat(dataRebalanced, dataTesting, imgClass.features);
                    // //         imgClass.classNumber = eachClass;
                    // //         imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32FC1); // Training
                    // //         imgClass.trainOrTest = Scalar(1);
                    // //         imgClass.trainOrTest.resize(imgClass.features.size().height, 2); // Testing
                            
                    // //         rebalancedData.push_back(imgClass);
                    // //     }
                    // //     else{
                    // //         rebalancedData.push_back(artificialData[eachClass]);
                    // //     }
                    
                    // // }
                    // // // stringstream numberOfImages;
                    // // // numberOfImages.str("");
                    // // // numberOfImages << trainTestOverSampled.size().height;
                    // // // cout << "number of images: " << trainTestOverSampled.size().height << endl;
                    // // // string name = featuresDir+descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";
                    // // cout << "---------------------------------------------------------------------------------------" << endl;
                    // // cout << "Classification using SMOTE" << endl;
                    // // // cout << "Features vectors file: " << name.c_str() << endl;
                    // // cout << "---------------------------------------------------------------------------------------" << endl;

                    // // // FILE *arq = fopen(name.c_str(), "w+");
                    // // // int w, z;
                    // // // fprintf(arq,"%d %d\t%d\n", total.size().height, numClasses, total.size().width);  
                    // // // for (w = 0; w < total.size().height; w++) {
                    // // //     fprintf(arq,"%d\t%d\t%d\t", w, (int) newClasses.at<float>(w,0), (int) trainTestOverSampled.at<float>(w,0));  
                    // // //     for(z = 0; z < total.size().width; z++) {
                    // // //         fprintf(arq,"%.5f ", total.at<float>(w, z));
                    // // //     }
                    // // //     fprintf(arq,"\n");  
                    // // //}

                    // // c.classify(prob, 10, rebalancedData, csvSmote.c_str());


                //         /* Copy the feature data to minorityClass */
                //         imbalancedData.rowRange(start,numTraining).copyTo(minorityTraining);
                //         imbalancedData.rowRange(numTraining,end).copyTo(minorityTesting);

                //         // performSmote();
                        /* Amount of SMOTE % */
                        // // neighbors = amountSmote/100;
                        // neighbors = 5; //TODO: Fix amount to smote for more than one class
                        // cout << endl << "SMOTE: Synthetic Minority Over-sampling Technique" << endl;

                //         /* Over-sampling the minority class */
                //         if (amountSmote > 0){
                //             synthetic = s.smote(minorityTraining, amountSmote, neighbors);
                //             cout << "minoritaria de treino " << minorityTraining.size().height << endl;
                //             cout << "minoritaria de teste " << minorityTesting.size().height << endl;
                //             /* Concatenate the minority class with the synthetic */
                //             vconcat(minorityTraining, synthetic, minorityRebalanced);
                //             vconcat(minorityRebalanced, minorityTesting, minorityOverSampled);
                //             cout << "\nminority over " << minorityOverSampled.size().height << " minorityClass " << minorityTraining.size().height << " synthetic " << synthetic.size().height << endl;
                //             Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);
                //             /* Select the majority classes */
                //             cout << "imbalancedData " << imbalancedData.size().height << endl;
                //             imbalancedData.rowRange(end, imbalancedData.size().height).copyTo(majority);
                //             imbalancedClasses.rowRange(end, imbalancedData.size().height).copyTo(majorityClasses);

                //             if (start != 0){ // Copy the initial majority
                //                 cout << " start != 0 " << endl;
                //                 Mat majorityInitial, majorityClassesInitial;
                //                 imbalancedData.rowRange(0, start).copyTo(majorityInitial);
                //                 imbalancedClasses.rowRange(0, start).copyTo(majorityClassesInitial);
                //                 vconcat(majority, majorityInitial, majority);
                //                 vconcat(majorityClasses, majorityClassesInitial, majorityClasses);
                //             }

                //             cout << "\nmajority " << majority.size().height << " end " << end << " start " << start << " Amount " << amountSmote << endl;
                //             /* Concatenate the feature samples and classes */
                //             vconcat(minorityClasses, majorityClasses, newClasses);
                //             total.release();
                //             vconcat(minorityOverSampled, majority, total);
                //             pair <int, int> minSmote(smallerClass+1, minorityRebalanced.size().height);
                //             cout << "total " << total.size().height << endl;
                //             cout << "trainTest " << trainTestImbalanced.size().height << endl;
                //             Mat trainOrTestForStartMin, trainOrTestForEndMin, trainTestOverSampled, trainOrTestForMaj, trainTestMin;
                //             trainTestImbalanced.rowRange(start,numTraining).copyTo(trainOrTestForStartMin);
                //             trainTestImbalanced.rowRange(numTraining,end).copyTo(trainOrTestForEndMin);
                //             trainTestImbalanced.rowRange(end,trainTestImbalanced.size().height).copyTo(trainOrTestForMaj);
                //             Mat newTrainTest(synthetic.size().height, 1, CV_32FC1, 1);
                //             vconcat(trainOrTestForStartMin, newTrainTest, trainTestOverSampled);
                //             vconcat(trainTestOverSampled, trainOrTestForEndMin, trainTestMin);
                //             trainTestOverSampled.release();
                //             vconcat(trainTestMin, trainOrTestForMaj, trainTestOverSampled);

                //             stringstream numberOfImages;
                //             numberOfImages.str("");
                //             numberOfImages << trainTestOverSampled.size().height;
                //             cout << "number of images: " << trainTestOverSampled.size().height << endl;
                //             string name = featuresDir+descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";
                //             cout << "---------------------------------------------------------------------------------------" << endl;
                //             cout << "Classification using SMOTE" << endl;
                //             cout << "Features vectors file: " << name.c_str() << endl;
                //             cout << "---------------------------------------------------------------------------------------" << endl;

                //             FILE *arq = fopen(name.c_str(), "w+");
                //             int w, z;
                //             fprintf(arq,"%d %d\t%d\n", total.size().height, numClasses, total.size().width);  
                //             for (w = 0; w < total.size().height; w++) {
                //                 fprintf(arq,"%d\t%d\t%d\t", w, (int) newClasses.at<float>(w,0), (int) trainTestOverSampled.at<float>(w,0));  
                //                 for(z = 0; z < total.size().width; z++) {
                //                     fprintf(arq,"%.5f ", total.at<float>(w, z));
                //                 }
                //                 fprintf(arq,"\n");  
                //             }

                //             c.bayes(prob, 10, total, newClasses, numClasses, minSmote, trainTestOverSampled, csvSmote.c_str());


                //             // FILE *arq = fopen("smote.data", "w+");
                //             // int w, z;
                //             // fprintf(arq,"%s\n", "DY");
                //             // fprintf(arq,"%d\n", newClasses.size().height);
                //             // fprintf(arq,"%d\n", total.size().width);
                //             // for(z = 0; z < total.size().width-1; z++) {
                //             //     fprintf(arq,"%s%d;", "attr",z);
                //             // }
                //             // fprintf(arq,"%s%d\n", "attr",z);
                //             // for (w = 0; w < newClasses.size().height; w++) {
                //             //     fprintf(arq,"%d%s;", w,".jpg");
                //             //     for(z = 0; z < total.size().width; z++) {
                //             //         fprintf(arq,"%.5f;", total.at<float>(w, z));
                //             //     }
                //             //     fprintf(arq,"%1.1f\n", newClasses.at<float>(w,0));  
                //             // }
                //             trainTest.release();
                //             trainOrTestForStartMin.release();
                //             trainOrTestForEndMin.release();
                //             trainTestOverSampled.release();
                //             trainOrTestForMaj.release();
                //             trainTestMin.release();
                //             minorityOverSampled.release();
                //             minorityClasses.release();
                //             majority.release();
                //             majorityClasses.release();
                //             newClasses.release();
                //             total.release();
                //             synthetic.release();
                //             minorityTraining.release();
                //             minorityTesting.release();
                //             minorityRebalanced.release();
                //         }
                //     }
                // }
            }
            data.clear();
        }
    }
    return 0;
}
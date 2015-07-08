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

Mat classifica(string dir, string features, string csvName, string csvOriginal, pair<int, int> min, int d, int m, int operation){

    Classifier c;
    Size size;
    int numClasses, generated = 0;
    float prob = 0.5;
    Mat data, classes, trainTest, trainTestOriginal;
    Artificial a;
    stringstream numImages;

    numImages.str("");
    numImages << min.second;
    // const char* numImg = numImages.str().c_str();
    string temp_str = numImages.str();
    char* numImg = (char*) temp_str.c_str();

    string originalDescriptor = desc(dir, features, d, m, "original");
    data = readFeatures(originalDescriptor, &classes, &trainTestOriginal, &numClasses);

    size = data.size();
    if (size.height != 0){
        /* Classifying without rebalancing */
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using desbalanced data" << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        c.bayes(prob, 10, data, classes, numClasses, min, trainTestOriginal, csvOriginal.c_str());
    }

    generated = a.generate(dir, 1, operation);
    min.second += generated;

    string fileDescriptor = desc(dir, features, d, m, "artificial");
    data = readFeatures(fileDescriptor, &classes, &trainTest, &numClasses);
    size = data.size();
    if (size.height != 0){
    	/* Classify with rebalanced data */
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using rebalanced data" << endl;
        cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        c.bayes(prob, 10, data, classes, numClasses, min, trainTest, csvName.c_str());
    }
    return trainTestOriginal;
}

Mat createDesbalancedFolders(int smallerClass, vector<int> *vectorRand, int d, int m, string csvName, string csvOriginal, int samples, int operation, string baseDir, string featuresDir, string newDir){

    int x;
    string dir, str, nameFile, name, nameDir;
    stringstream numImages, smallClass;
    Size size;
    ifstream myFile;
    Mat data, classes;
    pair <int, int> min(smallerClass, (*vectorRand).size()/2);

    numImages << (*vectorRand).size();
    smallClass << smallerClass;
    dir = newDir+numImages.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";  
    str += "cp -r "+baseDir+"* "+dir+";";
    str += "rm "+dir+smallClass.str()+"/*;";
    str += "mkdir -p "+dir+smallClass.str()+"/teste/ "+dir+smallClass.str()+"/treino/;";
    system(str.c_str());

    stringstream image;
    /* Copy some of the originals in treino vs teste */
    for(x = 0; x < (int)(*vectorRand).size()-1; x++){
        image.str("");
        image << (*vectorRand)[x];
        str = "cp "+baseDir+smallClass.str()+"/"+image.str()+".jpg ";
        str+= dir+smallClass.str()+"/treino/; ";
        x++;
        image.str("");
        image << (*vectorRand)[x];
        str += "cp "+baseDir+smallClass.str()+"/"+image.str()+".jpg ";
        str+= dir+smallClass.str()+"/teste/;";
        system(str.c_str());
    }
    /* Copy the rest of originals in teste */
    
    str = "bash "+newDir+"rename.sh "+dir+smallClass.str()+"/";
    system(str.c_str());
    Mat trainTestOriginal = classifica(dir, featuresDir, csvName, csvOriginal, min, d, m, operation);
    return trainTestOriginal;
}

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
pair <int, int> imbalance(Mat original, Mat classes, Mat *trainTest, int factor, int numClasses, Mat *imbalancedData, Mat *imbalancedClasses, int d, int m, string csvName, string csvOriginal, int operation, string baseDir, string featuresDir, string newDir){

    int total = 0, pos = 0, i, smallerClass, start, end, samples, num;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));
    Classifier c;
    c.findSmallerClass(classes, numClasses, &smallerClass, &start, &end);
    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    //samples = ceil(samples/factor);

    (*imbalancedData).create(num, size.width, CV_32FC1);
    (*imbalancedClasses).create(num, 1, CV_32FC1);

    while (total < ceil(samples/factor)) {
        /* Generate a random position to select samples to create the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = (*imbalancedData).row(total);
            original.row(pos).copyTo(tmp);
            (*imbalancedClasses).at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    (*trainTest) = createDesbalancedFolders(smallerClass+1, &vectorRand, d, m, csvName, csvOriginal, samples, operation, baseDir, featuresDir, newDir);

    /* Copy the majority data after the minority data */
    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            Mat tmp = (*imbalancedData).row(total);
            original.row(i).copyTo(tmp);
            (*imbalancedClasses).at<float>(total, 0) = classes.at<float>(i,0);
            total++;
        }
    }
    cout << "********* total " << total << endl;
    vectorRand.clear();
    pair <int, int> min(smallerClass+1, ceil(samples/factor)/2);
    return min;
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, smallerClass, amountSmote, start, end, neighbors, rep, m, d, operation;
    int initialMethod, endMethod;
    float prob = 0.5;
    ofstream csvFile;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir;
    Mat data, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced, trainTest;

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
    for (d = 1; d <= 7; d++){
        initialMethod = 1;
        endMethod = 4;
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
            data = readFeatures(fileDescriptor, &classes, &trainTest, &numClasses);
            size = data.size();
            if (size.height != 0){
                for (rep = 0; rep < 1; rep++){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using original vectors" << endl;
                    cout << "Features vectors file: " << name.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    pair <int, int> min(-1,-1);
                    c.bayes(prob, 10, data, classes, numClasses, min, trainTest, csvOriginal.c_str());
                    c.bayes(prob, 10, data, classes, numClasses, min, trainTest, csvSmote.c_str());
                    c.bayes(prob, 10, data, classes, numClasses, min, trainTest, csvRebalance.c_str());

                    c.findSmallerClass(classes, numClasses, &smallerClass, &start, &end);

                    //i = 2;
                    for (i = 2; (end-start)/2 >= 10; i*=2){

                        cout << "\n\n---------------------------------------------------------------------------------------" << endl;
                        cout << "Divide the number of original samples by a factor of " << i << " to create a minority class:"<< endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;

                        Mat imbalancedClasses, imbalancedData;
                        /* Desbalancing Data */
                        min = imbalance(data, classes, &trainTest, i, numClasses, &imbalancedData, &imbalancedClasses, d, m, csvRebalance, csvOriginal, operation, baseDir, featuresDir, newDir);

                        c.findSmallerClass(imbalancedClasses, numClasses, &smallerClass, &start, &end);
                        /* Copy the feature data to minorityClass */
                        imbalancedData.rowRange(start,min.second).copyTo(minorityTraining);
                        imbalancedData.rowRange(min.second,end).copyTo(minorityTesting);
                        cout << "minorityTesting size: " << minorityTesting.size().height << " from " << min.second << " until " << end << endl;
                        // performSmote();
                        /* Amount of SMOTE % */
                        // amountSmote = ((imbalancedData.size().height-end-(end-start)) / (end-start))*100.0;
                        amountSmote = ((100-(end-start)) / (end-start))*100.0;
                        neighbors = amountSmote/100;
                        // neighbors = 5; //TODO: Fix amount to smote for more than one class
                        cout << endl << "SMOTE: Synthetic Minority Over-sampling Technique" << endl;
                        cout << "Amount to SMOTE: " << amountSmote << "%" << endl;
                        /* Over-sampling the minority class */
                        if (amountSmote > 0){
                            neighbors = amountSmote/100;
                            synthetic = s.smote(minorityTraining, amountSmote, neighbors);
                            cout << "minoritaria de treino " << minorityTraining.size().height << endl;
                            cout << "minoritaria de teste " << minorityTesting.size().height << endl;
                            /* Concatenate the minority class with the synthetic */
                            vconcat(minorityTraining, synthetic, minorityRebalanced);
                            vconcat(minorityRebalanced, minorityTesting, minorityOverSampled);
                            cout << "\nminority over " << minorityOverSampled.size().height << " minorityClass " << minorityTraining.size().height << " synthetic " << synthetic.size().height << endl;
                            Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);
                            /* Select the majority classes */
                            cout << "imbalancedData " << imbalancedData.size().height << endl;
                            imbalancedData.rowRange(end, imbalancedData.size().height).copyTo(majority);
                            imbalancedClasses.rowRange(end, imbalancedData.size().height).copyTo(majorityClasses);

                            if (start != 0){ // Copy the initial majority
                                cout << " start != 0 " << endl;
                                Mat majorityInitial, majorityClassesInitial;
                                imbalancedData.rowRange(0, start).copyTo(majorityInitial);
                                imbalancedClasses.rowRange(0, start).copyTo(majorityClassesInitial);
                                vconcat(majority, majorityInitial, majority);
                                vconcat(majorityClasses, majorityClassesInitial, majorityClasses);
                            }

                            cout << "\nmajority " << majority.size().height << " end " << end << " start " << start << " Amount " << amountSmote << endl;
                            /* Concatenate the feature samples and classes */
                            vconcat(minorityClasses, majorityClasses, newClasses);
                            total.release();
                            vconcat(minorityOverSampled, majority, total);
                            pair <int, int> minSmote(smallerClass+1, minorityRebalanced.size().height);
                            cout << "total " << total.size().height << endl;
                            cout << "trainTest " << trainTest.size().height << endl;
                            Mat trainOrTestForStartMin, trainOrTestForEndMin, trainTestOverSampled, trainOrTestForMaj, trainTestMin;
                            trainTest.rowRange(start,min.second).copyTo(trainOrTestForStartMin);
                            trainTest.rowRange(min.second,end).copyTo(trainOrTestForEndMin);
                            trainTest.rowRange(end,trainTest.size().height).copyTo(trainOrTestForMaj);
                            Mat newTrainTest(synthetic.size().height, 1, CV_32FC1, 1);
                            vconcat(trainOrTestForStartMin, newTrainTest, trainTestOverSampled);
                            vconcat(trainTestOverSampled, trainOrTestForEndMin, trainTestMin);
                            trainTestOverSampled.release();
                            vconcat(trainTestMin, trainOrTestForMaj, trainTestOverSampled);

                            stringstream numberOfImages;
                            numberOfImages.str("");
                            numberOfImages << trainTestOverSampled.size().height;
                            cout << "number of images: " << trainTestOverSampled.size().height << endl;
                            string name = featuresDir+descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";
                            cout << "---------------------------------------------------------------------------------------" << endl;
                            cout << "Classification using SMOTE" << endl;
                            cout << "Features vectors file: " << name.c_str() << endl;
                            cout << "---------------------------------------------------------------------------------------" << endl;

                            FILE *arq = fopen(name.c_str(), "w+");
                            int w, z;
                            fprintf(arq,"%d %d\t%d\n", total.size().height, numClasses, total.size().width);  
                            for (w = 0; w < total.size().height; w++) {
                                fprintf(arq,"%d\t%d\t%d\t", w, (int) newClasses.at<float>(w,0), (int) trainTestOverSampled.at<float>(w,0));  
                                for(z = 0; z < total.size().width; z++) {
                                    fprintf(arq,"%.5f ", total.at<float>(w, z));
                                }
                                fprintf(arq,"\n");  
                            }

                            c.bayes(prob, 10, total, newClasses, numClasses, minSmote, trainTestOverSampled, csvSmote.c_str());


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
                            trainTest.release();
                            trainOrTestForStartMin.release();
                            trainOrTestForEndMin.release();
                            trainTestOverSampled.release();
                            trainOrTestForMaj.release();
                            trainTestMin.release();
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
            }
            data.release();
        }
    }
    return 0;
}
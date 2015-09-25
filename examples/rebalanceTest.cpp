/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalanceTest.h"

// string descriptorMethod[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
// string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

string desc(string dir, string features, int d, int m, string id){

    vector<int> paramCCV = {25};
    vector<int> paramACC = {1, 3, 5, 7};
    vector<int> parameters;
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramCCV, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramACC, 0, m, id);
    else
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, parameters, 0, m, id);
}

/* Generate a imbalanced class */
string imbalance(string database, string newDir, double prob, double id){

    int pos = 0, samples, imagesTraining, i, imgsInClass;
    int x, qtdClasses;
    string dir;
    vector<int> vectorRand, objperClass;
    srand(time(0));
    string str, nameFile, name, nameDir, directory;
    stringstream numImages, classNumber, image, globalFactor;
    Size size;
    ifstream myFile;
    Mat data, classes;
    double fator = 1.0;

    globalFactor << id;
    dir = newDir+"/steps-"+globalFactor.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r "+database+"* "+dir+";";
    //cout << " Executa " << str.c_str() << endl;
    system(str.c_str());

    directory = database+"/";
    qtdClasses = qtdArquivos(directory);

    for(i = 0; i < qtdClasses; i++) {
        classNumber.str("");
        classNumber << i;
        directory = database + "/" + classNumber.str()  + "/";
        imgsInClass = qtdArquivos(directory);
        if (imgsInClass == 0){
            fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
            exit(-1);
        }
        samples = imgsInClass*fator;
        if (samples > 0){

            numImages.str("");
            numImages << vectorRand.size();

            //str = "rm "+dir+classNumber.str()+"/*;";
            str = "rm -f -r "+dir+classNumber.str()+"/*;";
            str += "mkdir -p "+dir+classNumber.str()+"/treino/;";
            str += "mkdir -p "+dir+classNumber.str()+"/teste/;";

            //cout << " Executa " << str.c_str() << endl;
            system(str.c_str());

            /* Generate a random position to select samples to create the minority class */
            while ((int)vectorRand.size() < samples) {
                pos = rand() % imgsInClass;
                if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                    vectorRand.push_back(pos);
                }
            }

            /* Copy some of the originals to a training folder */
            imagesTraining = vectorRand.size()*prob;
            for(x = 0; x < imagesTraining; x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/treino/; ";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            /* Copy the rest of originals to a testing folder */
            for(x = imagesTraining; x < (int)vectorRand.size(); x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/teste/;";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            str = "bash "+newDir+"rename.sh "+dir+classNumber.str()+"/";
            //cout << " Executa " << str.c_str() << endl;
            system(str.c_str());

            vectorRand.clear();
        }
        fator -= id/(double)qtdClasses;
    }
    return dir;
}

vector<Classes> performSmote(vector<Classes> imbalancedData, int operation, int *total){

    int majority = -1, majorityClass = -1, eachClass, amountSmote;
    int numTraining = 0, numTesting = 0, i, x, neighbors, pos;
    vector<int> trainingNumber(imbalancedData.size(), 0);
    std::vector<Classes>::iterator it;
    vector<Classes> rebalancedData;
    Mat synthetic;
    SMOTE s;

    for (it = imbalancedData.begin(); it != imbalancedData.end(); ++it){
        if (it->fixedTrainOrTest){
            for(i = 0; i < it->trainOrTest.size().height; ++i){
                if (it->trainOrTest.at<int>(i,0) == 1)
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

    (*total) = 0;
    for (eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass){

        Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_64FC1);
        Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_64FC1);

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
                if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 1){
                    dataTraining.resize(numTraining+1);
                    Mat tmp = dataTraining.row(numTraining);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTraining++;
                }
                if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 2){
                    dataTesting.resize(numTesting+1);
                    Mat tmp = dataTesting.row(numTesting);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTesting++;
                }
            }

            if (operation != 0){
                synthetic = s.smote(dataTraining, amountSmote, neighbors);
            }
            else {
                synthetic.create(amountSmote, imbalancedData[eachClass].features.size().width, CV_64FC1);
                for (x = 0; x < amountSmote; x++){
                    pos = rand() % (dataTraining.size().height);
                    Mat tmp = synthetic.row(x);
                    dataTraining.row(pos).copyTo(tmp);
                }
            }

            cout << " new synthetic samples: " << amountSmote << endl;

            /* Concatenate original with synthetic data*/
            Classes imgClass;
            Mat dataRebalanced;
            vconcat(dataTraining, synthetic, dataRebalanced);
            vconcat(dataRebalanced, dataTesting, imgClass.features);
            imgClass.classNumber = eachClass;
            imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32S); // Training
            imgClass.trainOrTest = Scalar::all(1);
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
    Size size;
    Artificial a;
    ofstream csvFile;
    stringstream numImages, globalFactor;
    int numClasses, m, d, operation, h, w, totalRebalanced;
    double prob = 0.5;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
    string directory, str, bestDir;
    vector<Classes> imbalancedData, artificialData;
    vector<int> objperClass;
    vector<vector<double> > rebalancedFscore, desbalancedFscore;

    if (argc != 8){
        cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
        cout << "\t(0) Directory to place tests\n" << endl;
        cout << "\t(1) Image Directory\n" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        cout << "\t(3) Analysis Directory\n" << endl;
        cout << "\t(4) Descriptor Method:\n" << endl;
        cout << "\t\t1-BIC  2-GCH  3-CCV  4-Haralick  5-ACC  6-LBP  7-HOG  8-Contour  9-Fisher" << endl << endl;
        cout << "\t(5) Quantization Method:\n" << endl;
        cout << "\t\t1-Intensity  2-Luminance  3-Gleam  4-MSB" << endl << endl;
        cout << "\t(6) Artificial Generation Operation - Use 1 for ALL\n" << endl;
        cout << "\t./rebalanceTest Desbalanced/ Desbalanced/original/ Desbalanced/features/ Desbalanced/analysis/ 0\n" << endl;
        exit(-1);
    }
    newDir = string(argv[1]);
    baseDir = string(argv[2]);
    featuresDir = string(argv[3]);
    analysisDir = string(argv[4]);
    operation = atoi(argv[5]);
    string op = argv[5];
    d = atoi(argv[6]);
    m = atoi(argv[7]);

    /*  Available
            descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
            Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    double factor = 1.0;

    /* Desbalancing Data */
    cout << "\n\n------------------------------------------------------------------------------------" << endl;
    cout << "Divide the number of original samples to create a minority class:" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    string dirImbalanced = imbalance(baseDir, newDir, 0.5, factor);

    /* Generate Artificial Images */
    string newDirectory = dirImbalanced+"/../Rebalanced";
    string dirRebalanced = a.generate(dirImbalanced, newDirectory, operation);

    csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
    csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
    csvSmote = analysisDir+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
    csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";

    /* Feature extraction from images */
    string originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "desbalanced");
    /* Read the feature vectors */
    imbalancedData = ReadFeaturesFromFile(originalDescriptor);
    numClasses = imbalancedData.size();
    if (numClasses != 0){
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using desbalanced data" << endl;
        cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        desbalancedFscore = c.classify(prob, 1, imbalancedData, csvDesbalanced.c_str(), factor);
        imbalancedData.clear();
    }

    string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
    artificialData = ReadFeaturesFromFile(fileDescriptor);
    if (artificialData.size() != 0){
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using rebalanced data" << endl;
        cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        rebalancedFscore = c.classify(prob, 1, artificialData, csvRebalance.c_str(), factor);
        artificialData.clear();
    }

    /* Generate Synthetic SMOTE samples */
    imbalancedData = ReadFeaturesFromFile(originalDescriptor);
    vector<Classes> rebalancedData = performSmote(imbalancedData, operation, &totalRebalanced);
    if (rebalancedData.size() != 0){
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using SMOTE" << endl;
        cout << "Features vectors file: " << name.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        c.classify(prob, 1, rebalancedData, csvSmote.c_str(), factor);

        stringstream numberOfImages;
        numberOfImages.str("");
        numberOfImages << totalRebalanced;
        string name = featuresDir+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_256c_100r_"+numberOfImages.str()+"i_smote.csv";

        FILE *arq = fopen(name.c_str(), "w+");
        fprintf(arq,"%d %d\t%d\n", totalRebalanced, (int)rebalancedData.size(), rebalancedData[0].features.size().width);
        int imgNumber = 0;
        for(std::vector<Classes>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
            for (h = 0; h < it->features.size().height; h++, imgNumber++){
                fprintf(arq,"%d\t%d\t%d\t", imgNumber, it->classNumber, it->trainOrTest.at<int>(h,0));
                for (w = 0; w < it->features.size().width; w++){
                    fprintf(arq,"%.5f ", it->features.at<float>(h, w));
                }
                fprintf(arq,"\n");
            }
        }
        rebalancedData.clear();
    }
    return 0;
}

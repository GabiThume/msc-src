/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalanceTest.h"

// string descriptorMethod[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
// string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

// descriptor(databaseDir.c_str(), descMethodDir.c_str(), descMethod, colors, resize, normalization, params, numParameters, deleteNull, quantMethod, id.c_str());
string desc(string dir, string features, int d, int m, string id){

    vector<int> paramCCV = {25};
    vector<int> paramACC = {1, 3, 5, 7};
    vector<int> parameters;
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return descriptor(dir, features, d, 64, 1, 1, paramCCV, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return descriptor(dir, features, d, 64, 1, 1, paramACC, 0, m, id);
    else
        return descriptor(dir, features, d, 64, 1, 1, parameters, 0, m, id);
}

/* Generate a imbalanced class */
string imbalance(string database, string newDir, double prob, double id){

    int pos = 0, samples, imagesTraining, i, imgsInClass;
    int x, qtdClasses, qtdImgTotal, maxc;
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
    dir = newDir+"/Imbalance-"+globalFactor.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r "+database+"* "+dir+";";
    system(str.c_str());

    directory = database+"/";
    qtdClasses = qtdArquivos(directory);
    qtdImgTotal = qtdImagensTotal(database, qtdClasses, &objperClass, &maxc);

    for(i = 0; i < qtdClasses; i++) {
        classNumber.str("");
        classNumber << i;
        directory = database + "/" + classNumber.str()  + "/";
        imgsInClass = qtdArquivos(directory);
        if (imgsInClass == 0){
            fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
            exit(-1);
        }
        samples = ceil(imgsInClass*fator);
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
                str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
                str+= dir+classNumber.str()+"/treino/; ";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            /* Copy the rest of originals to a testing folder */
            for(x = imagesTraining; x < (int)vectorRand.size(); x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
                str+= dir+classNumber.str()+"/teste/;";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            str = "bash scripts/rename.sh "+dir+classNumber.str()+"/";
            system(str.c_str());

            vectorRand.clear();
        }
        fator -= id/(double)qtdClasses;
    }
    return dir;
}

string performSmote(vector<Classes> imbalancedData, int operation, string csvSmote){

    int majority = -1, majorityClass = -1, eachClass, amountSmote, countImg = 0;
    int numTraining = 0, numTesting = 0, i, x, neighbors, pos, total, h, w;
    vector<int> trainingNumber(imbalancedData.size(), 0);
    std::vector<Classes>::iterator it;
    vector<Classes> rebalancedData;
    stringstream numberOfImages;
    ofstream arq;
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
            trainingNumber[it->classNumber] = it->trainOrTest.size().height/2.0;
        }
        if (trainingNumber[it->classNumber] > majority){
            majorityClass = it->classNumber;
            majority = trainingNumber[it->classNumber];
        }
    }

    total = 0;
    for (eachClass = 0; eachClass < (int) imbalancedData.size(); ++eachClass){

        Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_32FC1);
        Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_32FC1);

        numTraining = 0;
        numTesting = 0;

        cout << "In class " << eachClass << " original training images: " << trainingNumber[eachClass] << endl;
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
                synthetic.create(amountSmote, imbalancedData[eachClass].features.size().width, CV_32FC1);
                for (x = 0; x < amountSmote; x++){
                    pos = rand() % (dataTraining.size().height);
                    Mat tmp = synthetic.row(x);
                    dataTraining.row(pos).copyTo(tmp);
                }
            }

            cout << "SMOTE generated " << amountSmote << " new synthetic samples" << endl;

            /* Concatenate original with synthetic data*/
            Classes imgClass;
            Mat dataRebalanced;
            vconcat(dataTraining, synthetic, dataRebalanced);
            vconcat(dataRebalanced, dataTesting, imgClass.features);
            imgClass.classNumber = eachClass;
            imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32S); // Training
            imgClass.trainOrTest = Scalar::all(1);
            imgClass.trainOrTest.resize(imgClass.features.size().height, Scalar::all(2)); // Testing

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

    numberOfImages.str("");
    numberOfImages << total;

    string name = csvSmote + "256c_100r_"+numberOfImages.str()+"i_smote.csv";
    arq.open(name.c_str(), ios::out);
    if (!arq.is_open()) {
      cout << "It is not possible to open the feature's file: " << name << endl;
      exit(-1);
    }
    cout << "---------------------------------------------------------------------------------------" << endl;
    cout << "Wrote on data file named " << name << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    arq << total << '\t' << rebalancedData.size() << '\t' << rebalancedData[0].features.size().width << endl;
    for(std::vector<Classes>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
        cout << "SMOTE >>>>>>>>>> " << it->features.size() << endl;
        for (h = 0; h < it->features.size().height; h++){
            arq << countImg << '\t' << it->classNumber << '\t' << it->trainOrTest.at<int>(h,0) << '\t';
            for (w = 0; w < it->features.size().width; w++){
              arq << it->features.at<float>(h, w) << " ";
            }
            arq << endl;
            countImg++;
        }
    }
    arq.close();
    return name;
}

int main(int argc, char const *argv[]){

    Classifier c;
    Size size;
    Artificial a;
    ofstream csvFile;
    stringstream numImages, globalFactor;
    int numClasses, m, d, operation, i, h, w, totalRebalanced, indexDescriptor, minoritySize;
    double prob = 0.5, fscoreMean, bestFscoreMean;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
    string directory, str, bestDir, op;
    vector<Classes> imbalancedData, artificialData, originalData, rebalancedData;
    vector<int> objperClass;
    vector<vector<double> > rebalancedFscore, desbalancedFscore;
    bool copyBestFscoreToOtherFolder = false;

    if (argc != 3){
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
    // featuresDir = string(argv[3]);
    // analysisDir = string(argv[4]);
    // d = atoi(argv[5]);
    // m = atoi(argv[6]);
    // operation = atoi(argv[7]);
    // op = argv[7];

    /*  Available
            descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
            Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
    */
    double factor = 1.6;
    // vector <int> descriptors {1, 6, 7};
    vector <int> descriptors {1, 2, 3, 4, 5, 6, 7, 8};

    /* Desbalancing Data */
    cout << "\n\n------------------------------------------------------------------------------------" << endl;
    cout << "Divide the number of original samples to create a minority class:" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    string dirImbalanced = imbalance(baseDir, newDir, 0.5, factor);

    for (operation = 0; operation <= 8; operation++){

        stringstream operationstr;
        operationstr << operation;
        op = operationstr.str();
        /* Generate Artificial Images */
        string newDirectory = newDir+"/Rebalanced-"+op;
        string dirRebalanced = a.generate(dirImbalanced, newDirectory, operation);

        for (indexDescriptor = 0; indexDescriptor < descriptors.size(); indexDescriptor++){
            d = descriptors[indexDescriptor];

            // if (d == 1)
            //     m = 4;
            // else
            //     m = 2;
            for (m = 1; m <= 4; m++){
                csvOriginal = newDir+"/analysis/"+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                csvDesbalanced = newDir+"/analysis/"+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                csvSmote = newDir+"/analysis/"+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                csvRebalance = newDir+"/analysis/"+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                featuresDir = dirRebalanced+"/../features/";

                /* Feature extraction from images */
                string originalDescriptor = desc(baseDir, featuresDir, d, m, "original");
                /* Read the feature vectors */
                originalData = readFeatures(originalDescriptor);
                numClasses = originalData.size();
                if (numClasses != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using original data" << endl;
                    cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.findSmallerClass(originalData, &minoritySize);
                    c.classify(prob, 1, originalData, csvOriginal.c_str(), minoritySize);
                    originalData.clear();
                }

                /* Feature extraction from images */
                originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "desbalanced");
                /* Read the feature vectors */
                imbalancedData = readFeatures(originalDescriptor);
                numClasses = imbalancedData.size();
                if (numClasses != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using desbalanced data" << endl;
                    cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.findSmallerClass(imbalancedData, &minoritySize);
                    desbalancedFscore = c.classify(prob, 1, imbalancedData, csvDesbalanced.c_str(), minoritySize);
                    imbalancedData.clear();
                }

                string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
                artificialData = readFeatures(fileDescriptor);
                if (artificialData.size() != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using rebalanced data" << endl;
                    cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.findSmallerClass(artificialData, &minoritySize);
                    rebalancedFscore = c.classify(prob, 1, artificialData, csvRebalance.c_str(), minoritySize);
                    artificialData.clear();
                }

                /* Generate Synthetic SMOTE samples */
                imbalancedData = readFeatures(originalDescriptor);
                string descSmote = newDirectory+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                string smoteDescriptor = performSmote(imbalancedData, operation, descSmote);
                rebalancedData = readFeatures(smoteDescriptor);
                if (rebalancedData.size() != 0){
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    cout << "Classification using SMOTE" << endl;
                    cout << "Features vectors file: " << smoteDescriptor.c_str() << endl;
                    cout << "---------------------------------------------------------------------------------------" << endl;
                    c.findSmallerClass(rebalancedData, &minoritySize);
                    c.classify(prob, 1, rebalancedData, csvSmote.c_str(), minoritySize);
                    rebalancedData.clear();
                }
            }
        }
    }
    return 0;
}

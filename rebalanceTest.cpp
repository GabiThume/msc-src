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
string imbalance(string database, string newDir, double prob){

    int pos = 0, samples, imagesTraining, i, imgsInClass;
    string dir;
    vector<int> vectorRand, objperClass;
    srand(time(0));
    int x, qtdClasses, qtdImgTotal, maxc;
    string str, nameFile, name, nameDir, directory;
    stringstream numImages, classNumber, image;
    Size size;
    ifstream myFile;
    Mat data, classes;
    double fator = 1.0;

    dir = newDir+"/steps/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r "+database+"* "+dir+";";
    //cout << " Executa " << str.c_str() << endl;
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

        samples = imgsInClass*fator;
        if (samples > 0){

            numImages.str("");
            numImages << vectorRand.size();

            //str = "rm "+dir+classNumber.str()+"/*;";
            str = "rm -f -r "+dir+classNumber.str()+"/*;";
            str += "mkdir -p "+dir+classNumber.str()+"/treino/;";
            str += "mkdir -p "+dir+classNumber.str()+"/teste/;";

            //str = "mkdir -p "+dir+classNumber.str()+"/teste/ "+dir+classNumber.str()+"/treino/;";
            //cout << " Executa " << str.c_str() << endl;
            system(str.c_str());

            /* Generate a random position to select samples to create the minority class */
            while ((int)vectorRand.size() < samples) {
                pos = rand() % imgsInClass;
                if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                    vectorRand.push_back(pos);
               }
            }

            // dir = newDir+numImages.str()+"/";
            // str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
            // str += "cp -r "+database+"* "+dir+";";
            // str += "rm "+dir+classNumber.str()+"/*;";
            // str += "mkdir -p "+dir+classNumber.str()+"/teste/ "+dir+classNumber.str()+"/treino/;";
            // cout << " Executa " << str.c_str() << endl;
            //system(str.c_str());

            /* Copy some of the originals in treino vs teste */
            imagesTraining = vectorRand.size()*prob;
            for(x = 0; x < imagesTraining; x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/treino/; ";
                system(str.c_str());
                //cout << " Executa " << str.c_str() << endl;
            }

            for(x = imagesTraining; x < (int)vectorRand.size(); x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".jpg ";
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
        //fator -= 0.1;
        fator -= 1.0/(double)qtdClasses;
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

// string fileDescriptor = desc(baseDir, featuresDir, d, m, "original");
// data = readFeatures(fileDescriptor);
// numClasses = data.size();
// if (numClasses != 0){
    // for (rep = 0; rep < 1; rep++){
        // cout << "---------------------------------------------------------------------------------------" << endl;
        // cout << "Classification using original vectors" << endl;
        // cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
        // cout << "---------------------------------------------------------------------------------------" << endl;
        // c.classify(prob, 10, data, csvOriginal.c_str(), 1);
        // c.classify(prob, 10, data, csvSmote.c_str(), 1);
        // c.classify(prob, 10, data, csvRebalance.c_str(), 1);

int main(int argc, char const *argv[]){

    Classifier c;
    Size size;
    Artificial a;
    ofstream csvFile;
    stringstream numImages;
    int numClasses, rep, m, d, operation, i, initialMethod, endMethod, h, w, totalRebalanced;
    double prob = 0.5, fscoreMean, bestFscoreMean;
    string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
    string directory, str, bestDir;
    vector<Classes> imbalancedData, artificialData;
    vector<int> objperClass;
    vector<vector<double> > rebalancedFscore, desbalancedFscore;
    vector<double> fscores, bestFscore, desbalancedFscores;

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

    for (rep = 0; rep < 10; rep ++){
        /* Desbalancing Data */
        cout << "\n\n------------------------------------------------------------------------------------" << endl;
        cout << "Divide the number of original samples to create a minority class:" << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        string dirImbalanced = imbalance(baseDir, newDir, 0.5);

        for (operation = 1; operation <= 15; operation ++){

            stringstream operationstr;
            operationstr << operation;
            op = operationstr.str();
            /* Generate Artificial Images */
            string dirRebalanced = a.generate(dirImbalanced, operation);
            /*  Available
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
                    string originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "desbalanced");
                    /* Read the feature vectors */
                    imbalancedData = readFeatures(originalDescriptor);
                    numClasses = imbalancedData.size();
                    if (numClasses != 0){
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using desbalanced data" << endl;
                        cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        desbalancedFscore = c.classify(prob, 1, imbalancedData, csvDesbalanced.c_str(), 2);
                    }

                    string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
                    artificialData = readFeatures(fileDescriptor);
                    if (artificialData.size() != 0){
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using rebalanced data" << endl;
                        cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        rebalancedFscore = c.classify(prob, 1, artificialData, csvRebalance.c_str(), 2);
                    }

                    /* Generate Synthetic SMOTE samples */
                    vector<Classes> rebalancedData = performSmote(imbalancedData, operation, &totalRebalanced);
                    if (rebalancedData.size() != 0){
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        cout << "Classification using SMOTE" << endl;
                        cout << "Features vectors file: " << name.c_str() << endl;
                        cout << "---------------------------------------------------------------------------------------" << endl;
                        c.classify(prob, 1, rebalancedData, csvSmote.c_str(), 2);

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
                    }

                    for (i = 0; i < (int) rebalancedFscore.size(); i++){
                		fscores.push_back(c.calculateMean(rebalancedFscore[i]));
                        desbalancedFscores.push_back(c.calculateMean(desbalancedFscore[i]));
                        if (bestFscore.size() < fscores.size()){
                            bestFscore.push_back(0);
                            bestFscoreMean = 0;
                        }
                	}
                    fscoreMean = c.calculateMean(fscores);
                    if (fscoreMean > bestFscoreMean){
                        bestFscoreMean = fscoreMean;
                        bestDir = newDir+"/BestMeanFscore/";
                        str = "rm -f -r "+bestDir+"/*;";
                        str += "mkdir -p "+bestDir+";";
                        str += "cp -R "+dirRebalanced+"/* "+bestDir+";";
                        str += "cp -R "+csvSmote+"FScore.csv "+bestDir+";";
                        str += "cp -R "+csvRebalance+"FScore.csv "+bestDir+";";
                        str += "cp -R "+csvDesbalanced+"FScore.csv "+bestDir+";";
                        str += "cp -R "+csvSmote+"BalancedAccuracy.csv "+bestDir+";";
                        str += "cp -R "+csvRebalance+"BalancedAccuracy.csv "+bestDir+";";
                        str += "cp -R "+csvDesbalanced+"BalancedAccuracy.csv "+bestDir+";";
                        system(str.c_str());
                    }
                    for (i = 0; i < (int) rebalancedFscore.size(); i++){
                        /* If it is a better generation for some class */
                        double diff = fscores[i] - desbalancedFscores[i];
                        cout << "DIFF " << diff <<  endl;
                        if (diff > bestFscore[i]){
                            bestFscore[i] = diff;
                            stringstream classe;
                            classe << i;
                            bestDir = newDir+"/BestFscore/Generation_"+classe.str()+"/";
                        	str = "rm -f -r "+bestDir+"/*;";
                        	str += "mkdir -p "+bestDir+";";
                        	str += "cp -R "+dirRebalanced+"/* "+bestDir+";";
                            str += "cp -R "+csvSmote+"FScore.csv "+bestDir+";";
                            str += "cp -R "+csvRebalance+"FScore.csv "+bestDir+";";
                            str += "cp -R "+csvDesbalanced+"FScore.csv "+bestDir+";";
                            str += "cp -R "+csvSmote+"BalancedAccuracy.csv "+bestDir+";";
                            str += "cp -R "+csvRebalance+"BalancedAccuracy.csv "+bestDir+";";
                            str += "cp -R "+csvDesbalanced+"BalancedAccuracy.csv "+bestDir+";";
                            cout << "Copy generation of class " << i << " to " << bestDir << endl;
                            // cout << str << endl;
                        	system(str.c_str());
                        }
                    }
                    fscores.clear();
                    desbalancedFscores.clear();
                }
            }
        }
    }
    return 0;
}

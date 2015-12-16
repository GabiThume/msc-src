/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "rebalanceTest.h"

// std::string descriptorMethod[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
// std::string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

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

/* Generate a imbalanced class */
std::string imbalance(std::string database, std::string newDir, double prob, double id){

    int pos = 0, samples, imagesTraining, i, imgsInClass, x, qtdClasses;
    std::string dir;
    std::vector<int> vectorRand, objperClass;
    srand(time(0));
    std::string str, nameFile, name, nameDir, directory;
    std::stringstream numImages, classNumber, image, globalFactor;
    cv::Size size;
    std::ifstream myFile;
    cv::Mat data, classes;
    double fator = 1.0;

    globalFactor << id;
    dir = newDir+"/steps-"+globalFactor.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
    str += "cp -r "+database+"* "+dir+";";
    //std::cout << " Executa " << str.c_str() << std::endl;
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

            //std::cout << " Executa " << str.c_str() << std::endl;
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
                //std::cout << " Executa " << str.c_str() << std::endl;
            }

            /* Copy the rest of originals to a testing folder */
            for(x = imagesTraining; x < (int)vectorRand.size(); x++){
                image.str("");
                image << vectorRand[x];
                str = "cp "+database+classNumber.str()+"/"+image.str()+".jpg ";
                str+= dir+classNumber.str()+"/teste/;";
                system(str.c_str());
                //std::cout << " Executa " << str.c_str() << std::endl;
            }

            str = "bash "+newDir+"rename.sh "+dir+classNumber.str()+"/";
            //std::cout << " Executa " << str.c_str() << std::endl;
            system(str.c_str());

            vectorRand.clear();
        }
        fator -= id/(double)qtdClasses;
    }
    return dir;
}

std::string performSmote(std::vector<ImageClass> imbalancedData, int operation, std::string csvSmote){

    int majority = -1, majorityClass = -1, eachClass, amountSmote, countImg = 0;
    int numTraining = 0, numTesting = 0, i, x, neighbors, pos, total, h, w;
    std::vector<int> trainingNumber(imbalancedData.size(), 0);
    std::vector<ImageClass>::iterator it;
    std::vector<ImageClass> rebalancedData;
    std::stringstream numberOfImages;
    std::ofstream arq;
    cv::Mat synthetic;
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

        cv::Mat dataTraining(0, imbalancedData[eachClass].features.size().width, CV_32FC1);
        cv::Mat dataTesting(0, imbalancedData[eachClass].features.size().width, CV_32FC1);

        numTraining = 0;
        numTesting = 0;

        std::cout << "In class " << eachClass << " original training images: " << trainingNumber[eachClass] << std::endl;
        /* Find out how many samples are needed to rebalance */
        amountSmote = trainingNumber[majorityClass] - trainingNumber[eachClass];

        if (amountSmote > 0){
            //neighbors = 5;
            neighbors = (double)trainingNumber[majorityClass]/(double)trainingNumber[eachClass];
            //std::cout << " neighbors " << neighbors << std::endl;

            for (x = 0; x < imbalancedData[eachClass].trainOrTest.size().height; ++x){
                if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 1){
                    dataTraining.resize(numTraining+1);
                    cv::Mat tmp = dataTraining.row(numTraining);
                    imbalancedData[eachClass].features.row(x).copyTo(tmp);
                    numTraining++;
                }
                if (imbalancedData[eachClass].trainOrTest.at<int>(x,0) == 2){
                    dataTesting.resize(numTesting+1);
                    cv::Mat tmp = dataTesting.row(numTesting);
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
                    cv::Mat tmp = synthetic.row(x);
                    dataTraining.row(pos).copyTo(tmp);
                }
            }

            std::cout << "SMOTE generated " << amountSmote << " new synthetic samples" << std::endl;

            /* Concatenate original with synthetic data*/
            Classes imgClass;
            cv::Mat dataRebalanced;
            vconcat(dataTraining, synthetic, dataRebalanced);
            vconcat(dataRebalanced, dataTesting, imgClass.features);
            imgClass.classNumber = eachClass;
            imgClass.trainOrTest.create(dataRebalanced.size().height, 1, CV_32S); // Training
            imgClass.trainOrTest = cv::Scalar::all(1);
            imgClass.trainOrTest.resize(imgClass.features.size().height, cv::Scalar::all(2)); // Testing

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

    std::string name = csvSmote + "256c_100r_"+numberOfImages.str()+"i_smote.csv";
    arq.open(name.c_str(), std::ios::out);
    if (!arq.is_open()) {
      std::cout << "It is not possible to open the feature's file: " << name << std::endl;
      exit(-1);
    }
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Wrote on data file named " << name << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    arq << total << '\t' << rebalancedData.size() << '\t' << rebalancedData[0].features.size().width << std::endl;
    for(std::vector<ImageClass>::iterator it = rebalancedData.begin(); it != rebalancedData.end(); ++it) {
        std::cout << "SMOTE >>>>>>>>>> " << it->features.size() << std::endl;
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
    return name;
}

int main(int argc, char const *argv[]){

    Classifier c;
    cv::Size size;
    Artificial a;
    std::ofstream csvFile;
    std::stringstream numImages, globalFactor;
    int numClasses, m, d, operation, i, x, minoritySize;
    double prob = 0.5, fscoreMean, bestFscoreMean;
    std::string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
    std::string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
    std::string directory, str, bestDir;
    std::vector<ImageClass> imbalancedData, artificialData, rebalancedData;
    std::vector<int> objperClass;
    std::vector<std::vector<double> > rebalancedFscore, desbalancedFscore;
    std::vector<double> fscores, bestFscore, desbalancedFscores;
    bool copyBestFscoreToOtherFolder = true;

    if (argc != 8){
        std::cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << std::endl;
        std::cout << "\t(0) Directory to place tests\n" << std::endl;
        std::cout << "\t(1) Image Directory\n" << std::endl;
        std::cout << "\t(2) Features Directory\n" << std::endl;
        std::cout << "\t(3) Analysis Directory\n" << std::endl;
        std::cout << "\t(4) Descriptor Method\n" << std::endl;
        std::cout << "\t\t1 - BIC\t2 - GCH\t3 - CCV\t4 - Haralick\t5 - ACC\t6 - LBP\t7 - HOG\t8 - Contour Extraction\t9 - Fisher Vectors" << std::endl;
        std::cout << "\t(5) Quantization Method\n" << std::endl;
        std::cout << "\t\t1 - Intensity\t2 - Luminance\t3 - Gleam\t4 - MSB" << std::endl;
        std::cout << "\t(6) Artificial Generation Operation - Use 1 for ALL\n" << std::endl;
        std::cout << "\t./rebalanceTest Desbalanced/ Desbalanced/original/ Desbalanced/features/ Desbalanced/analysis/ 0\n" << std::endl;
        exit(-1);
    }
    newDir = std::string(argv[1]);
    baseDir = std::string(argv[2]);
    featuresDir = std::string(argv[3]);
    analysisDir = std::string(argv[4]);
    operation = atoi(argv[5]);
    std::string op = argv[5];
    d = atoi(argv[6]);
    m = atoi(argv[7]);

    /*  Available
            descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
            Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
    */

    std::vector<double> factors = {1.0, 1.5, 1.75};

    for (x = 0; x < (int)factors.size(); x++){
        /* Desbalancing Data */
        std::cout << "\n\n------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Divide the number of original samples to create a minority class:" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::string dirImbalanced = imbalance(baseDir, newDir, 0.5, factors[x]);
        // std::string dirImbalanced = imbalance(baseDir, newDir, 0.5, 0.5);
        // imbalance(baseDir, newDir, 0.5, 0.25);
        for (operation = 1; operation <= 15; operation ++){
        //
        //     std::stringstream operationstr;
        //     operationstr << operation;
        //     op = operationstr.str();
            /* Generate Artificial Images */
            globalFactor << factors[x];
            std::string newDirectory = dirImbalanced+"/../Rebalanced-"+globalFactor.str();
            std::string dirRebalanced = a.generate(dirImbalanced, newDirectory, operation);
            for (d = 1; d <= 8; d++){
                for (m = 1; m <= 4; m++){
                    csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                    csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                    csvSmote = analysisDir+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                    csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";

                    /* Feature extraction from images */
                    std::string originalDescriptor = desc(dirImbalanced, featuresDir, d, m, "desbalanced");
                    /* Read the feature vectors */
                    imbalancedData = ReadFeaturesFromFile(originalDescriptor);
                    numClasses = imbalancedData.size();
                    if (numClasses != 0){
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        std::cout << "Classification using desbalanced data" << std::endl;
                        std::cout << "Features vectors file: " << originalDescriptor.c_str() << std::endl;
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        std::cout << "X: " << x << "ID: " << factors[x] << std::endl;
                        desbalancedFscore = c.classify(prob, 1, imbalancedData, csvDesbalanced.c_str(), x);
                        imbalancedData.clear();
                    }

                    std::string fileDescriptor = desc(dirRebalanced, featuresDir, d, m, "artificial");
                    artificialData = ReadFeaturesFromFile(fileDescriptor);
                    if (artificialData.size() != 0){
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        std::cout << "Classification using rebalanced data" << std::endl;
                        std::cout << "Features vectors file: " << fileDescriptor.c_str() << std::endl;
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        rebalancedFscore = c.classify(prob, 1, artificialData, csvRebalance.c_str(), x);
                        artificialData.clear();
                    }

                    /* Generate Synthetic SMOTE samples */
                    imbalancedData = ReadFeaturesFromFile(originalDescriptor);
                    std::string descSmote = newDirectory+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
                    std::string smoteDescriptor = performSmote(imbalancedData, operation, descSmote);
                    rebalancedData = ReadFeaturesFromFile(smoteDescriptor);
                    if (rebalancedData.size() != 0){
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        std::cout << "Classification using SMOTE" << std::endl;
                        std::cout << "Features vectors file: " << smoteDescriptor.c_str() << std::endl;
                        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                        c.findSmallerClass(rebalancedData, &minoritySize);
                        c.classify(prob, 1, rebalancedData, csvSmote.c_str(), minoritySize);
                        rebalancedData.clear();
                    }

                    if (copyBestFscoreToOtherFolder){

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
                            std::cout << "DIFF " << diff <<  std::endl;
                            if (diff > bestFscore[i]){
                                bestFscore[i] = diff;
                                std::stringstream classe;
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
                                std::cout << "Copy generation of class " << i << " to " << bestDir << std::endl;
                                // std::cout << str << std::endl;
                            	system(str.c_str());
                            }
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

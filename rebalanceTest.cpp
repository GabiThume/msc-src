/**
 *
 *  Author: Gabriela Thumé
 *  Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"
#include "artificialGeneration.h"

void desc(string dir, string features, int d, int m, string id){

    int paramCCV[1] = {25};
    int paramACC[4] = {1, 3, 5, 7};

    /* If descriptor ==  CCV, threshold is required */
    if (d == 3){
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramCCV, 1, 0, m, id.c_str());
    }
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5){
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, paramACC, 4, 0, m, id.c_str());
    }
    else{
        descriptor(dir.c_str(), features.c_str(), d, 256, 1, 1, 0, 0, 0, m, id.c_str());
    }
}

void classifica(string dir, string features, string csvName, pair<int, int> min, int d, int m, int operation){

    Classifier c;
    Size size;
    int numClasses, generated;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir, expected, expectedACC;
    Mat data, classes;
    Artificial a;
    stringstream numImages;
    numImages.str("");
    numImages << min.second;
    // const char* numImg = numImages.str().c_str();
    string temp_str = numImages.str();
    char* numImg = (char*) temp_str.c_str();

    string methods[4] = {"Intensity", "Gleam", "Luminance", "MSB"};
    string descriptors[5] = {"BIC", "GCH", "CCV", "Haralick6", "ACC"};

    generated = a.generate(dir, 1, operation);
    min.second += generated;

    desc(dir, features, d, m, numImg);

    nameDir = features + "/";
    directory = opendir(nameDir.c_str());
    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            expected = descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_"+numImages.str()+".txt";
            expectedACC = descriptors[d-1]+"_"+methods[m-1]+"_256c_4d_100r_"+numImages.str()+".txt";
            if(nameFile == expected || nameFile == expectedACC){
                name = nameDir + arq->d_name;
                myFile.open(name.c_str());

                // Read the feature vectors 
                data = readFeatures(name.c_str(), classes, numClasses);
                size = data.size();
                if (size.height != 0){
                	/* Classify with rebalanced data */
                    cout << "Classification with rebalanced data " << data.size().height << endl;
                    c.bayes(prob, 10, data, classes, numClasses, min, csvName.c_str());
                }
            }
        }
    }
}

void createDesbalancedFolders(int smallerClass, vector<int> &vectorRand, int d, int m, string csvName, int samples, int operation){

    int x;
    string dir, str, nameFile, name, nameDir;
    stringstream numImages, smallClass;
    Size size;
    ifstream myFile;
    Mat data, classes;
    pair <int, int> min(smallerClass, vectorRand.size()/2);

    numImages << vectorRand.size();
    smallClass << smallerClass;
    dir = "Desbalanced/"+numImages.str()+"/";
    str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";  
    str += "cp -r Desbalanced/original/* "+dir+";";
    str += "rm "+dir+smallClass.str()+"/*;";
    str += "mkdir -p "+dir+smallClass.str()+"/teste/ "+dir+smallClass.str()+"/treino/;";
    system(str.c_str());

    stringstream image;
    /* Copy some of the originals in treino vs teste */
    for(x = 0; x < (int)vectorRand.size()-1; x++){
        image.str("");
        image << vectorRand[x];
        str = "cp Desbalanced/original/"+smallClass.str()+"/"+image.str()+".jpg ";
        str+= dir+smallClass.str()+"/treino/; ";
        x++;
        image.str("");
        image << vectorRand[x];
        str += "cp Desbalanced/original/"+smallClass.str()+"/"+image.str()+".jpg ";
        str+= dir+smallClass.str()+"/teste/;";
        system(str.c_str());
    }
    /* Copy the rest of originals in teste */
    
    str = "bash Desbalanced/rename.sh "+dir+smallClass.str()+"/";
    system(str.c_str());
    
    classifica(dir, "features/artificial/", csvName, min, d, m, operation);
}

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
pair <int, int> imbalance(Mat original, Mat classes, int factor, int numClasses, Mat &imbalancedData, Mat &imbalancedClasses, int d, int m, string csvName, int operation){

    int total = 0, pos = 0, i, smallerClass, start, end, samples, num;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));
    Classifier c;

    c.findSmallerClass(classes, numClasses, smallerClass, start, end);

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    //samples = ceil(samples/factor);

    imbalancedData.create(num, size.width, CV_32FC1);
    imbalancedClasses.create(num, 1, CV_32FC1);

    while (total < ceil(samples/factor)) {
        /* Generate a random position to select samples to create the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = imbalancedData.row(total);
            original.row(pos).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    createDesbalancedFolders(smallerClass+1, vectorRand, d, m, csvName, samples, operation);

    /* Copy the majority data after the minority data */
    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            Mat tmp = imbalancedData.row(total);
            original.row(i).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(i,0);
            total++;
        }
    }
    vectorRand.clear();
    pair <int, int> min(smallerClass+1, ceil(samples/factor)/2);
    return min;
}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, smallerClass, amountSmote, start, end, neighbors, rep, m, d, operation;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    ofstream csvFile;
    string nameFile, name, nameDir, baseDir, featuresDir, analysis, descriptorName, method;
    string csvOriginal, csvSmote, csvRebalance;
    Mat data, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;
    Mat minorityTraining, minorityTesting, minorityRebalanced;

    if (argc != 4){
        cout << "\nUsage: ./rebalanceTest (1) (2) (3)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        cout << "\t(3) Artificial Generation Operation - Use 0 for ALL\n" << endl;
        exit(-1);
    }
    baseDir = string(argv[1]);
    featuresDir = string(argv[2]);
    operation = atoi(argv[3]);

    string descriptors[5] = {"BIC", "GCH", "CCV", "Haralick6", "ACC"};
    string methods[4] = {"Intensity", "Gleam", "Luminance", "MSB"};
    string op = argv[3];
    // MSB para cor e intensity para haralick
    for (d = 1; d < 6; d++){
        if (d == 4)
            m = 1;
        else
            m = 4;
        //for (m = 1; m < 5; m++){
            csvOriginal = "Desbalanced/analysis/"+op+"-original_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvSmote = "Desbalanced/analysis/"+op+"-smote_"+descriptors[d-1]+"_"+methods[m-1]+"_";
            csvRebalance = "Desbalanced/analysis/"+op+"-artificial_"+descriptors[d-1]+"_"+methods[m-1]+"_";

            // csvFile.open((csvOriginal+"BalancedAccuracy.csv").c_str(), ios::trunc);
            // csvFile.close();
            // csvFile.open((csvSmote+"BalancedAccuracy.csv").c_str(), ios::trunc);
            // csvFile.close();
            // csvFile.open((csvRebalance+"BalancedAccuracy.csv").c_str(), ios::trunc);
            // csvFile.close();
            // csvFile.open((csvOriginal+"FScore.csv").c_str(), ios::trunc);
            // csvFile.close();
            // csvFile.open((csvSmote+"FScore.csv").c_str(), ios::trunc);
            // csvFile.close();
            // csvFile.open((csvRebalance+"FScore.csv").c_str(), ios::trunc);
            // csvFile.close();

            /* Feature extraction from images */
            desc(baseDir, featuresDir, d, m, "");

            nameDir = string(featuresDir.c_str()) + "/";
            directory = opendir(nameDir.c_str());

            if (directory != NULL){
                while ((arq = readdir(directory))){

                    nameFile = arq->d_name;
                    if (nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_100r_.txt" ||
                        nameFile == descriptors[d-1]+"_"+methods[m-1]+"_256c_4d_100r_.txt"){
                        name = nameDir + arq->d_name;
                        myFile.open(name.c_str());

                        /* Read the feature vectors */
                        data = readFeatures(name.c_str(), classes, numClasses);
                        size = data.size();
                        if (size.height != 0){
                            for (rep = 0; rep < 20; rep++){
                                cout << "---------------------------------------------------------------------------------------" << endl;
                                cout << "Classification using original vectors" << endl;
                                cout << "Features vectors file: " << name.c_str() << endl;
                                cout << "---------------------------------------------------------------------------------------" << endl;
                                pair <int, int> min(-1,-1);
                                c.bayes(prob, 10, data, classes, numClasses, min, csvOriginal.c_str());
                                c.bayes(prob, 10, data, classes, numClasses, min, csvSmote.c_str());
                                c.bayes(prob, 10, data, classes, numClasses, min, csvRebalance.c_str());
                                c.findSmallerClass(classes, numClasses, smallerClass, start, end);
                                //i = 8;
                                for (i = 2; (end-start)/2 >= 10; i*=2){

                                    cout << "\n\n---------------------------------------------------------------------------------------" << endl;
                                    cout << "Divide the number of original samples by a factor of " << i << " to create a minority class:"<< endl;
                                    cout << "---------------------------------------------------------------------------------------" << endl;

                                    Mat imbalancedClasses, imbalancedData;
                                    /* Desbalancing Data */
                                    min = imbalance(data, classes, i, numClasses, imbalancedData, imbalancedClasses, d, m, csvRebalance, operation);
                                    /* Classifying without rebalancing */
	                                cout << "Classification without rebalancing" << endl;
                                    cout << "total original " << imbalancedData.size().height << endl;
                                    c.bayes(prob, 10, imbalancedData, imbalancedClasses, numClasses, min, csvOriginal.c_str());

                                    c.findSmallerClass(imbalancedClasses, numClasses, smallerClass, start, end);
                                    /* Copy the feature data to minorityClass */
                                    imbalancedData.rowRange(start,min.second).copyTo(minorityTraining);
                                    imbalancedData.rowRange(min.second,end).copyTo(minorityTesting);
                                    /* Amount of SMOTE % */
                                    amountSmote = ((imbalancedData.size().height-end-(end-start)) / (end-start))*100.0;
                                    //amountSmote = 100;
                                    neighbors = amountSmote/100;
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
                                    /* Select the majority classes */
                                    imbalancedData.rowRange(end, imbalancedData.size().height).copyTo(majority);
                                    imbalancedClasses.rowRange(end, imbalancedData.size().height).copyTo(majorityClasses);

                                    if (start != 0){ // Copy the initial majority
                                        Mat majorityInitial, majorityClassesInitial;
                                        imbalancedData.rowRange(0, start).copyTo(majorityInitial);
                                        imbalancedClasses.rowRange(0, start).copyTo(majorityClassesInitial);
                                        vconcat(majority, majorityInitial, majority);
                                        vconcat(majorityClasses, majorityClassesInitial, majorityClasses);
                                    }

                                    //cout << "\nmajority " << majority.size().height << " end " << end << " start " << start << " Amount " << amountSmote << endl;

                                    /* Concatenate the feature samples and classes */
                                    vconcat(minorityClasses, majorityClasses, newClasses);
                                    vconcat(minorityOverSampled, majority, total);
                                    pair <int, int> minSmote(smallerClass+1, minorityRebalanced.size().height);
                                    cout << "total p smote " << total.size().height << endl;
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
                        myFile.close();
                    }
                }
            }
            data.release();
        //}
    }
    return 0;
}
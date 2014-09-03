/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "smote.h"

/* Read the features of the file and save them in Mat data */
Mat readFeatures(const string& filename, Mat &classes, int &nClasses){

    int i, j;
    float features;
    Mat data;
    size_t n, d;
    ifstream myFile(filename.c_str());
    string line, infos, numImage, classe, numFeatures, numClasses, objetos;

    if(!myFile)
        throw exception();

    /* Read the first line, which contains the number of objects, classes and features */
    getline(myFile, infos);
    if (infos == "")
        return Mat();
    stringstream info(infos);
    getline(info, objetos, '\t');
    getline(info, numClasses, '\t');
    nClasses = atoi(numClasses.c_str());
    getline(info, numFeatures, '\t');

    n = atoi(objetos.c_str());
    d = atoi(numFeatures.c_str());

    /* Create a Mat named data with the file data provided */
    data.create(n, d, CV_32FC1);
    classes.create(n, 1, CV_32FC1);
    while (getline(myFile, line)) {
        stringstream vector_features(line);
        getline(vector_features, numImage, '\t');
        getline(vector_features, classe, '\t');
        i = atoi(numImage.c_str());
        j = 0;
        while(vector_features >> features) {
            data.at<float>(i, j) = (float)features;
            j++;
        }
        classes.at<float>(i, 0)=atoi(classe.c_str());
    }

    myFile.close();
    return data;
}

/* Find which is the smaller class and where it starts and ends */
void findSmallerClass(Mat classes, int numClasses, int &smallerClass, int &start, int &end){

    int i, smaller;
    Size size = classes.size();
    vector<int> dataClasse(numClasses, 0);

    /* Discover the number of samples for each class */
    for(i = 0; i < size.height; i++){
        dataClasse[classes.at<float>(i,0)-1]++;
    }

    /* Find out which is the minority class */
    smaller = size.height +1;
    smallerClass = -1;
    for(i = 0; i < (int) dataClasse.size(); i++){
        if(dataClasse[i] < smaller){
            smaller = dataClasse[i];
            smallerClass = i;
        }
    }

    /* Where the minority class starts and ends */
    start = -1;
    end = -1;
    for(i = 0; i < size.height; i++){
        if(classes.at<float>(i,0)-1 == smallerClass){
            if (start == -1){
                start = i;
            }
        }
        else if (start != -1){
            end = i;
            break;
        }
    }

}

/* Generate a imbalanced class and save it in imbalancedData and imbalancedClasses */
void imbalance(Mat original, Mat classes, int factor, int numClasses, Mat &imbalancedData, Mat &imbalancedClasses){

    int total = 0, pos = 0, i, smallerClass, start, end, samples, num;
    Size size = original.size();
    vector<int> vectorRand;
    Mat other, otherClasses;
    srand(time(0));

    findSmallerClass(classes, numClasses, smallerClass, start, end);

    samples = end - start;
    num = size.height - samples + ceil(samples/factor);
    samples = ceil(samples/factor);

    imbalancedData.create(num, size.width, CV_32FC1);
    imbalancedClasses.create(num, 1, CV_32FC1);

    while (total < samples) {
        /* Generate a random position to select samples to crete the minority class */
        pos = start + (rand() % end);
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
            vectorRand.push_back(pos);
            Mat tmp = imbalancedData.row(total);
            original.row(pos).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(start,0);
            total++;
       }
    }

    for (i = end; i < size.height; i++) {
        if (!count(vectorRand.begin(), vectorRand.end(), i)){
            Mat tmp = imbalancedData.row(total);
            original.row(i).copyTo(tmp);
            imbalancedClasses.at<float>(total, 0) = classes.at<float>(i,0);
            total++;
        }
    }

}

int main(int argc, char const *argv[]){

    SMOTE s;
    Classifier c;
    Size size;
    int numClasses, i, smallerClass, amountSmote, start, end, neighbors;
    float prob = 0.5;
    DIR *directory;
    struct dirent *arq;
    ifstream myFile;
    string nameFile, name, nameDir;
    Mat data, minorityClass, classes, minorityOverSampled, majority, majorityClasses, newClasses, total, synthetic;

    if (argc != 3){
        cout << "\nUsage: ./smoteTest (1) (2)\n\n\t(1) Image Directory" << endl;
        cout << "\t(2) Features Directory\n" << endl;
        exit(-1);
    }

    /* Feature extraction from images */
    descriptor(argv[1], argv[2], 4, 256, 1, 0, 0, 0, 0, 4);

    nameDir = string(argv[1]) + "/";
    directory = opendir(nameDir.c_str());

    if (directory != NULL){
        while ((arq = readdir(directory))){

            nameFile = arq->d_name;
            name = nameDir + arq->d_name;
            myFile.open(name.c_str());

            /* Read the feature vectors */
            data = readFeatures(name.c_str(), classes, numClasses);
            size = data.size();

            if (size.height != 0){

                cout << "---------------------------------------------------------------" << endl;
                cout << endl << "Features vectors file: " << name.c_str() << endl << endl;
                cout << "---------------------------------------------------------------" << endl;
                cout << "Classification using original vectors" << endl;
                c.bayes(data, classes, numClasses, prob, 10);

                for (i = 2; i <= 10; i*=2){

                    cout << "---------------------------------------------------------------" << endl;
                    cout << endl <<  "Normal Bayes Classification for imbalanced classes" << endl << endl;
                    cout << "\tDivide the number of original samples by a factor of " << i << endl <<"\tto create a minority class:"<< endl;

                    Mat imbalancedClasses, imbalancedData;
                    imbalance(data, classes, i, numClasses, imbalancedData, imbalancedClasses);
                    c.bayes(imbalancedData, imbalancedClasses, numClasses, prob, 10);
                    size = imbalancedData.size();
                    findSmallerClass(imbalancedClasses, numClasses, smallerClass, start, end);

                    /* Copy the feature data to minorityClass */
                    imbalancedData.rowRange(start,end).copyTo(minorityClass);
                    /* Amount of SMOTE % */
                    amountSmote = 100;
                    neighbors = 5;
                    /* Over-sampling the minority class */
                    synthetic = s.smote(minorityClass, amountSmote, neighbors);

                    /* Concatenate the minority class with the synthetic */
                    vconcat(minorityClass, synthetic, minorityOverSampled);
                    Mat minorityClasses(minorityOverSampled.size().height, 1, CV_32FC1, smallerClass+1);

                    /* Select the majority classes */
                    imbalancedData.rowRange(end, size.height).copyTo(majority);
                    imbalancedClasses.rowRange(end, size.height).copyTo(majorityClasses);

                    /* Concatenate the feature samples and classes */
                    vconcat(minorityClasses, majorityClasses, newClasses);
                    vconcat(minorityOverSampled, majority, total);

                    cout << endl << "\tSMOTE: Synthetic Minority Over-sampling Technique" << endl;
                    cout << "\tAmount to SMOTE: " << amountSmote << "%" << endl;
                    c.bayes(total, newClasses, numClasses, prob, 10);

                    minorityOverSampled.release();
                    minorityClasses.release();
                    majority.release();
                    majorityClasses.release();
                    newClasses.release();
                    total.release();
                }
            }
            myFile.close();
           }
    }
    return 0;
}
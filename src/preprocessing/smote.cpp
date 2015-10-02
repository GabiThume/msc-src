/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 *  Based on paper's "SMOTE: Synthetic Minority Over-sampling Technique" pseudo-code
 **/

#include "preprocessing/smote.h"

/* Generate the synthetic samples to over-sample the minority class */
void SMOTE::populate(Mat minority, Mat neighbors, Mat *synthetic, int *index, int amountSmote, int i, int nearestNeighbors){

    int attributes, nn, attr;
    double gap, dif, attrOriginal, neighbor;
    vector<int> vectorRand;
    Size s = minority.size();
    attributes = s.width;

    /* amountSmote is how many synthetic samples need to be generated for i */
    while(amountSmote != 0){
        /* Choose randomly one of the nearest neighbors of i */
        nn = 1 + (rand() % nearestNeighbors);

        if (!count(vectorRand.begin(), vectorRand.end(), nn)){
            vectorRand.push_back(nn);

            for(attr = 0; attr < attributes; attr++){
                neighbor = neighbors.at<float>(i, nn);
                attrOriginal = minority.at<float>(i, attr);

                /* The difference between the feature vector under and its
                nearest neighbor*/
                dif = minority.at<float>(neighbor, attr) - attrOriginal;
                /* Multiply this difference with a number between 0 and 1 */
                gap = (double) rand()/(RAND_MAX);
                /* Add it to the original feature vector */
                (*synthetic).at<float>((*index), attr) = attrOriginal + gap*dif;
            }
            (*index)++;
            amountSmote = amountSmote - 1;
        }

    }
}

/* Compute the nearest neighbors */
void SMOTE::computeNeighbors(Mat minority, int nearestNeighbors, Mat *neighbors){

    Size s = minority.size();
    int i, k, max, index = 0;
    Mat classes(s.height, 1, CV_32FC1);

    for(i = 0; i < s.height; i++) {
        classes.at<float>(i,0) = index;
        index++;
    }

    KNearest knn(minority, classes);

    k = nearestNeighbors+1;
    max = knn.get_max_k();
    knn.find_nearest(minority, (k > max ? max : k), 0, 0, neighbors, 0);

    classes.release();
}

/* Synthetic Minority Over-sampling Technique */
Mat SMOTE::smote(Mat minority, int amountSmote, int nearestNeighbors){

    Size s = minority.size();
    int i, samples, pos, index = 0;
    int minoritySamples = s.height;
    int attributes = s.width;
    vector<int> vectorRand;
    Mat newMinority;
    srand(time(NULL));

    if(amountSmote == 0)
        return Mat();

    /* If amount to smote is less than 100%, randomize the minority class
    samples as only a random percent of them will be SMOTEd */
    if(amountSmote < s.height){

        //minoritySamples = (amountSmote/100.0)*minoritySamples;
        minoritySamples = amountSmote;
        newMinority.create(minoritySamples, attributes, CV_32FC1);
        samples = 0;
        while (samples < minoritySamples) {
            /* Generate a random position for the minority class samples */
            pos = rand() % (s.height);
            if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                vectorRand.push_back(pos);
                Mat tmp = newMinority.row(samples);
                minority.row(pos).copyTo(tmp);
                samples++;
           }
        }
        minority = newMinority;

        // amountSmote = s.height;
    }

    /* If amount to smote is higher than 100%, randomize the minority class
    samples as a random percent of them will be SMOTEd */
    if(amountSmote > s.height){
        minoritySamples = amountSmote;
        newMinority.create(minoritySamples, attributes, CV_32FC1);
        samples = 0;
        while (samples < minoritySamples) {
            /* Generate a random position for the minority class samples */
            pos = rand() % (s.height);
            Mat tmp = newMinority.row(samples);
            minority.row(pos).copyTo(tmp);
            samples++;
        }
        minority = newMinority;
    }

    //amountSmote = amountSmote/100.0;
    Mat synthetic(amountSmote, attributes, CV_32FC1);
    // Mat synthetic(minoritySamples*amountSmote, attributes, CV_32FC1);
    Mat neighbors(minoritySamples, nearestNeighbors, CV_32FC1);

    /* Compute all the neighbors for the minority class */
    computeNeighbors(minority, nearestNeighbors, &neighbors);

    /* For each sample, generate it(s) synthetic(s) sample(s) */
    for(i = 0; i < minoritySamples; i++){
        populate(minority, neighbors, &synthetic, &index, 1, i, nearestNeighbors);
    }

    return synthetic;
}

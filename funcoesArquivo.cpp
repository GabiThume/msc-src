/**
 * 	Authors:
 *		Gabriela Thumé
 *  	Luciana Calixta Escobar
 *
 * 	Universidade de São Paulo / ICMC
 *
 **/

#include "funcoesArquivo.h"

/* Read the features and save them in Mat data */
vector<Classes> readFeatures(string filename){

    int j, newSize = 0;
    float features;
    size_t d;
    string line, infos, numImage, classe, trainTest, numFeatures, numClasses, objetos;
	vector<Classes> data;
    int previousClass = -1, actualClass;
    Classes imgClass;

    ifstream myFile(filename.c_str());
    if(!myFile)
        throw exception();

    /* Read the first line, which contains the number of objects, classes and features */
    getline(myFile, infos);
    if (infos == "")
        return Mat();
    stringstream info(infos);
    getline(info, objetos, '\t');
    getline(info, numClasses, '\t');
    //(*nClasses) = atoi(numClasses.c_str());
    getline(info, numFeatures, '\t');

    //n = atoi(objetos.c_str());
    d = atoi(numFeatures.c_str());

    while (getline(myFile, line)) {
        stringstream vector_features(line);
        getline(vector_features, numImage, '\t');
        getline(vector_features, classe, '\t');
        getline(vector_features, trainTest, '\t');
        actualClass = atoi(classe.c_str());
        if (previousClass != actualClass){

	    	if (previousClass != -1){
	    		data.push_back(imgClass);
	    	}
	    	previousClass = actualClass;
			imgClass.features.create(0, d, CV_32FC1);
			imgClass.trainOrTest.create(0, 1, CV_32FC1);
			imgClass.fixedTrainOrTest = false;
		}

		newSize = imgClass.features.size().height+1;
		imgClass.features.resize(newSize);
		imgClass.trainOrTest.resize(newSize);

        j = 0;
        while(vector_features >> features) {
			imgClass.features.at<float>(newSize-1,j) = (float) features;
            j++;
        }
        imgClass.trainOrTest.at<float>(newSize-1, 0)=atoi(trainTest.c_str());
	    imgClass.classNumber = actualClass;
	    if (atoi(trainTest.c_str())!=0)
	    	imgClass.fixedTrainOrTest = true;
    }
	if (previousClass != -1){
		data.push_back(imgClass);
	}

    myFile.close();
    return data;
}

int qtdArquivos(string directory){

 	int count = 0;
 	struct dirent *sDir = NULL;
 	DIR *dir = NULL;

 	dir = opendir(directory.c_str());
 	if(dir == NULL) {
 		return 0;
 	}

 	while((sDir = readdir(dir))) {
 		if ((strcmp(sDir->d_name, ".") 			!= 0) &&
 			(strcmp(sDir->d_name, "..") 		!= 0) &&
 			(strcmp(sDir->d_name, ".DS_Store") 	!= 0) &&
 			(strcmp(sDir->d_name, ".directory") != 0)) {
 			count++;
 		}
 	}

    closedir(dir);
    return count;
}

int qtdImagensTotal(string base, int qtdClasses, vector<int> *objClass, int *maxs){

	int i, count = 0, currentSize;
	string directory;
	*maxs = 0;

	for (i = 0; i < qtdClasses; i++){
		directory = base + "/" + to_string(i)  + "/treino/";
		currentSize = qtdArquivos(directory);
		directory = base + "/" + to_string(i)  + "/teste/";
		currentSize += qtdArquivos(directory);
		if (currentSize == 0){
			directory = base + "/" + to_string(i)  + "/";
			currentSize = qtdArquivos(directory);
			if (currentSize == 0){
		 		fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
		 	}
		}
		(*objClass).push_back(currentSize);
		count += currentSize;
		if (currentSize > *maxs || *maxs == 0)
			*maxs = currentSize;
	}

	return count;
}

string descriptor(string database, string featuresDir, int method, int colors, double resizeFactor, int normalization, int *param, int nparam, int deleteNull, int quantization, string id = ""){

	int i, j, k, numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0, grid;
	int resizingFactor = (int)(resizeFactor*100), maxc = 0, x;
	float min, max, normFactor;
	string nome, directory;
	double **coocurrenceMatrix = NULL;
	Mat img, featureVector, features, labels, trainTest, newimg;
	FILE *arq;
	string quantizationsNames[4] = {"Intensity", "Luminance", "Gleam", "MSB"};
	string descriptors[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
    vector<int> objperClass;

	cout << "\n---------------------------------------------------------------------------------------" << endl;
	cout << "Image feature extraction using " << descriptors[method-1] << " and " << quantizationsNames[quantization-1] << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	// Check how many classes and images there are
    directory = database+"/";
	qtdClasses = qtdArquivos(directory);
	qtdImgTotal = qtdImagensTotal(database, qtdClasses, &objperClass, &maxc);

	labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
	trainTest = Mat::zeros(qtdImgTotal, 1, CV_8U);

	for(i = 0; i < qtdClasses; i++) {

		directory = database + "/" + to_string(i)  + "/treino/";
		numImages = qtdArquivos(directory);
		treino = numImages;

		directory = database + "/" + to_string(i)  + "/teste/";
		numImages += qtdArquivos(directory);

		if (numImages == 0){
			directory = database + "/" + to_string(i)  + "/";
			numImages = qtdArquivos(directory);
			if (numImages == 0){
		 		fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
		 	}
		}

		cout << "class " << i << ": " << database + "/" + to_string(i) << " has " << numImages << " images" << endl;

		for(j = 0; j < numImages; j++)	{

            directory = database +"/"+to_string(i)+"/"+to_string(j);
			img = imread(directory+".jpg", CV_LOAD_IMAGE_COLOR);
			if (img.empty())
                img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
            trainTest.at<uchar>(imgTotal,0) = (uchar)0;
			if (img.empty()) {
                directory = database +"/"+to_string(i)+"/treino/"+to_string(j);
				img = imread(directory+".jpg", CV_LOAD_IMAGE_COLOR);
                if (img.empty())
                    img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
				trainTest.at<uchar>(imgTotal,0) = (uchar)1;
				if (img.empty()){
					directory = database +"/"+to_string(i)+"/teste/"+to_string(j-treino);
					img = imread(directory+".jpg", CV_LOAD_IMAGE_COLOR);
                    if (img.empty())
                        img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
					trainTest.at<uchar>(imgTotal,0) = (uchar)2;
					if (img.empty()){
						cout << "Error when trying to open an image of " << directory << endl;
						return "";
					}
				}
			}

			if (resizeFactor < 1) {
				resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
				if (imgTotal==0)
					imwrite("out.jpg", newimg);
			}
			else
				img.copyTo(newimg);

			switch(quantization){
				case 1:
                    QuantizationIntensity(newimg, &newimg, colors);
                    break;
				case 2:
                    QuantizationLuminance(newimg, &newimg, colors);
                    break;
				case 3:
                    QuantizationGleam(newimg, &newimg, colors);
                    break;
				case 4:
                    QuantizationMSB(newimg, &newimg, colors);
                    break;
				default:
                    cout << "Error: this quantization method does not exists." << endl;
                    exit(1);
			}

			switch(method){
	            /* BIC: image, descriptor, number of colors, normalization */
				case 1:
                    BIC(newimg, &featureVector, colors, normalization);
                    break;
			    /* GCH: image, descriptor, number of colors, normalization */
				case 2:
                    GCH(newimg, &featureVector, colors, normalization);
                    break;
			    /* CCV: image, descriptor, number of colors, normalization,
			          limiar coerente/incoerente */
				case 3:
                    CCV(newimg, &featureVector, colors, normalization, param[0]);
                    break;
			    /* HARALICK: image, co-occurrence matrix, descriptor
			               numero de cores, normalization */
				case 4:
                    coocurrenceMatrix = (double **)calloc(colors, sizeof(double*));
        			for (x = 0; x < colors; x++) {
        				coocurrenceMatrix[x]= (double *)calloc(colors, sizeof(double));
        			}
                    HARALICK(newimg, coocurrenceMatrix, &featureVector, colors, normalization);
                    break;
			    /* ACC: image, descriptor, number of colors, normalization,
			          distance vector, distance number */
				case 5:
                    ACC(newimg, &featureVector, colors, normalization, param, nparam);
                    break;
			    /* LBP */
				case 6:
                    LBP(newimg, &featureVector, colors);
                    break;
			    /* HOG */
				case 7:
					HOG(newimg, &featureVector);
                    break;
			    /* Contour */
				case 8:
                    contourExtraction(newimg, &featureVector);
                    break;
				case 9:
                    break;
				default:
                    cout << "Error: this description method does not exists." << endl;
                    exit(1);
			}

			labels.at<uchar>(imgTotal,0) = (uchar)i;
            if (features.size().height == 0){
                features = Mat::zeros(0, featureVector.size().width, CV_32F);
            }
            features.push_back(featureVector);
			imgTotal++;
            img.release();
            newimg.release();
            featureVector.release();
		}
	}

	if(method == 4 && normalization != 0){
		normFactor = (normalization == 1) ? 1.0 : 255.0;

		for(j = 0; j < features.cols; ++j){
			min = features.at<float>(0,j);
			max = features.at<float>(0,j);

			for(i = 1; i < features.rows; ++i){
				if(features.at<float>(i,j) > max)
					max = features.at<float>(i,j);
				if (features.at<float>(i,j) < min)
					min = features.at<float>(i,j);
			}
			for(i = 0; i < features.rows; ++i)
				features.at<float>(i,j) = normFactor * ((features.at<float>(i,j) - min) / (max - min));
		}
	}

    if (method != 5){
		nome = featuresDir+descriptors[method-1]+"_"+quantizationsNames[quantization-1];
        nome += "_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+to_string(qtdImgTotal)+"i_"+id+".csv";
    }
    else {
        nome = featuresDir+"/ACC_"+quantizationsNames[quantization-1]+"_"+to_string(colors);
        nome += "c_"+to_string(nparam)+"d_"+to_string(resizingFactor)+"r_"+to_string(qtdImgTotal)+"i_"+id+".csv";
    }

	arq = fopen(nome.c_str(), "w+");
	if (arq == 0){
		cout << "It is not possible to open the feature's file: " << nome << endl;
		return "";
	}

	fprintf(arq,"%d\t%d\t%d\n", qtdImgTotal, qtdClasses, features.cols);
	cout << "File: " << nome << endl;
	cout << "Objects: " << qtdImgTotal << " - Classes: " << qtdClasses << " - Features: " << features.cols << endl;
	for (i = 0; i < qtdClasses; i++) {
		int bars = (int) (((float) objperClass[i] / (float) qtdImgTotal)*50.0);
		cout << i << " ";
		for (j = 0; j < bars; j++){
			cout << "|";
		}
		float porc = (float)objperClass[i]/(float)qtdImgTotal;
		cout << " " << porc*100 << "%" << " (" << objperClass[i] << ")" <<endl;
	}

	for (i = 0; i < imgTotal; i++) {
	    // Write the image number and the referenced class
		fprintf(arq, "%d\t%d\t%d\t", i, labels.at<uchar>(i,0), trainTest.at<uchar>(i,0));
		for(k = 0; k < features.cols; k++) {
			if (normalization == 2)  {
				fprintf(arq,"%.f ", features.at<float>(i, k));
			}
			else
				fprintf(arq,"%.5f ", features.at<float>(i, k));
		}
		fprintf(arq,"\n");
	}

    bool writeDataFile = false;
    if (writeDataFile){

        cout << "---------------------------------------------------------------------------------------" << endl;
    	cout << "Wrote on data file named " << nome << endl;
    	cout << "---------------------------------------------------------------------------------------" << endl;
        FILE *arqVis = fopen((nome+"data").c_str(), "w+");
    	int w, z;
    	fprintf(arqVis,"%s\n", "DY");
    	fprintf(arqVis,"%d\n", labels.size().height);
    	fprintf(arqVis,"%d\n", features.size().width);
    	for(z = 0; z < features.size().width-1; z++) {
    	    fprintf(arqVis,"%s%d;", "attr",z);
    	}
    	fprintf(arqVis,"%s%d\n", "attr",z);
    	for (w = 0; w < labels.size().height; w++) {
    	    fprintf(arqVis,"%d%s;", w,".png");
    	    for(z = 0; z < features.size().width; z++) {
    	        fprintf(arqVis,"%.5f;", features.at<float>(w, z));
    	    }
    	    float numeroimg =  labels.at<uchar>(w,0);
    	    fprintf(arqVis,"%1.1f\n", numeroimg);
    	}
    }

	if (method == 4) {
		for (i = 0; i < colors; i++) {
			free(coocurrenceMatrix[i]);
		}
		free(coocurrenceMatrix);
	}

	fclose(arq);
	return nome;
}

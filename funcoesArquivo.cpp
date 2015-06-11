/**
 * 	Authors:
 *  	Luciana Calixta Escobar
 *		Gabriela Thumé
 *
 * 	Universidade de São Paulo / ICMC
 *  
 **/

#include "funcoesArquivo.h"

/* Read the features and save them in Mat data */
Mat readFeatures(string filename, Mat *classes, int *nClasses){

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
    (*nClasses) = atoi(numClasses.c_str());
    getline(info, numFeatures, '\t');

    n = atoi(objetos.c_str());
    d = atoi(numFeatures.c_str());

    /* Create a Mat named data with the file data provided */
    data.create(n, d, CV_32FC1);
    (*classes).create(n, 1, CV_32FC1);
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
        (*classes).at<float>(i, 0)=atoi(classe.c_str());
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
 			(strcmp(sDir->d_name, ".directory") != 0) &&
 			(strcmp(sDir->d_name, ".DS_Store") 	!= 0) &&
 			(strcmp(sDir->d_name, ".directory") != 0)) {
 			count++;
 		}
 	}

    closedir(dir);
    return count;
}

int qtdImagensTotal(string base, int qtdClasses, int *objClass, int *maxs){

	int i, count = 0, currentSize;
	string directory;
	*maxs = 0;

	for (i = 1; i <= qtdClasses; i++){
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
		objClass[i-1] = currentSize;
		count += currentSize;
		if (currentSize > *maxs || *maxs == 0)
			*maxs = currentSize;
	}

	return count;
}

int descriptor(string database, string featuresDir, int method, int colors, double resizeFactor, int normalization, int *param, int nparam, int deleteNull, int quantization, string id = ""){

	int i, j, k, numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0, grid;
	int featureVectorSize = 0, resizingFactor = (int)(resizeFactor*100), maxc = 0;
	float min, max, normFactor;
	string nome, directory;
	double **coocurrenceMatrix = NULL;
	Mat img, featureVector, features, labels, newimg;
	FILE *arq;
	string quantizationsNames[4] = {"Intensity", "Luminance", "Gleam", "MSB"};

	cout << "\n---------------------------------------------------------------------------------------" << endl;
	cout << "Image feature extraction" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	switch (method) {
		case 1: 
			nome = featuresDir+"/BIC_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			// sprintf(nome,"%s/BIC_%s_%dc_%dr_%s.txt", featuresDir, quantizationsNames[quantization-1], colors, resizingFactor, id);
			featureVectorSize = colors*2;
			cout << "BIC and " << quantizationsNames[quantization-1] << ":";
			break;
		case 2:
			nome = featuresDir+"/GCH_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = colors;
			cout << "GCH and " << quantizationsNames[quantization-1] << ":";
			break;
		case 3:
			nome = featuresDir+"/CCV_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = colors*2;
			cout << "CCV and " << quantizationsNames[quantization-1] << ":";
			break;
		case 4:
			nome = featuresDir+"/Haralick6_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			coocurrenceMatrix = (double **)calloc(colors, sizeof(double*));
			for (i=0; i<colors; i++) {
				coocurrenceMatrix[i]= (double *)calloc(colors, sizeof(double));
			}
			featureVectorSize = 6;
			cout << "Haralick-6 and " << quantizationsNames[quantization-1] << ":";
			break;
		case 5:
			nome = featuresDir+"/ACC_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(nparam)+"d_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = (colors*nparam);
			cout << "ACC and " << quantizationsNames[quantization-1] << " c/ " << nparam << " distancias : ";
			break;
		case 6:
			grid = 4;
			nome = featuresDir+"/LBP_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = 58*grid;
			cout << "LBP and " << quantizationsNames[quantization-1] << endl;
			break;
		case 7:
			nome = featuresDir+"/HOG_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = 58;
			cout << "HOG and " << quantizationsNames[quantization-1] << endl;
			break;
		case 8:
			nome = featuresDir+"/Contour_"+quantizationsNames[quantization-1]+"_"+to_string(colors)+"c_"+to_string(resizingFactor)+"r_"+id+".txt";
			featureVectorSize = 10;
			cout << "Contour and " << quantizationsNames[quantization-1] << endl;
			break;
		case 9:
			break;
		default:
			break;
	}

	// Allocates the feature vector of size featureVectorSize	
	featureVector.create(1, featureVectorSize, CV_64F);
	// Fill out with zeros
	featureVector = Scalar::all(0);
	
	arq = fopen(nome.c_str(), "w+");
	if (arq == 0){
		cout << "This feature's file does not exist." << endl;
		return -1;
	}
	directory = database+"/";
	
	// Check how many classes and images there are
	qtdClasses = qtdArquivos(directory);
	int *objperClass = (int *)malloc(qtdClasses*sizeof(int));
	qtdImgTotal = qtdImagensTotal(database, qtdClasses, objperClass, &maxc);
	fprintf(arq,"%d\t%d\t%d\n", qtdImgTotal, qtdClasses, featureVectorSize); 
	
	cout << colors << " colors, size " << resizeFactor << endl;
	cout << "File: " << nome << endl;
	cout << "Objects: " << qtdImgTotal << " - Classes: " << qtdClasses << " - Features: " << featureVectorSize << endl;

	for (i = 0; i < qtdClasses; i++) {
		int bars = (int) (((float) objperClass[i] / (float) qtdImgTotal)*50.0);
		cout << (i+1) << " ";
		for (j = 0; j < bars; j++){
			cout << "|";
		}	  
		float porc = (float)objperClass[i]/(float)qtdImgTotal;
		cout << " " << porc*100 << "%" << " (" << objperClass[i] << ")" <<endl;
	}

	// pos processa matriz de dados para retirar colunas nulas
	// para isso constroi matriz para armazenar vetores
 	//if (deleteNull == 1) {
	features = Mat::zeros(qtdImgTotal, featureVectorSize, CV_32F);
	labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
 	//}

	for(i = 1; i <= qtdClasses; i++) {

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

		cout << "class " << i << " : " << directory << " treino " << treino << " imagens " << numImages << endl;
		
		for(j = 0; j < numImages; j++)	{
			img = imread(database +"/"+to_string(i)+"/"+to_string(j)+".jpg", CV_LOAD_IMAGE_COLOR);
			if (img.empty()) { 
				img = imread(database +"/"+to_string(i)+"/treino/"+to_string(j)+".jpg", CV_LOAD_IMAGE_COLOR);
				if (img.empty()){
					directory = database +"/"+to_string(i)+"/teste/"+to_string(j-treino)+".jpg";
					img = imread(directory, CV_LOAD_IMAGE_COLOR);
					if (img.empty()){ 
						cout << "Error when trying to open an image of " << directory << endl; 
						return -1; 
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
                    QuantizationIntensity(&newimg, &newimg, colors);
                    break;
				case 2:
                    QuantizationLuminance(&newimg, &newimg, colors);
                    break;
				case 3:
                    QuantizationGleam(&newimg, &newimg, colors);
                    break;
				case 4:
                    QuantizationMSB(&newimg, &newimg, colors);
                    break;
				default: 
                    cout << "Error: this quantization method does not exists." << endl;
                    return -1;
			}
			
			switch(method){
	            /* BIC: image, descriptor, number of colors, normalization */
				case 1:
                    BIC(&newimg, &featureVector, colors, normalization);
                    break;
			    /* GCH: image, descriptor, number of colors, normalization */
				case 2:
                    GCH(&newimg, &featureVector, colors, normalization);
                    break;
			    /* CCV: image, descriptor, number of colors, normalization, 
			          limiar coerente/incoerente */
				case 3:
                    CCV(&newimg, &featureVector, colors, normalization, param[0]);
                    break;
			    /* HARALICK: image, co-occurrence matrix, descriptor
			               numero de cores, normalization */
				case 4:
                    HARALICK(&newimg, coocurrenceMatrix, &featureVector, colors, normalization);
                    break;
			    /* ACC: image, descriptor, number of colors, normalization,
			          distance vector, distance number */
				case 5:
                    ACC(&newimg, &featureVector, colors, normalization, param, nparam);
                    break;
			    /* LBP */
				case 6:
                    LBP(&newimg, &featureVector, colors);
                    break;
			    /* HOG */
				case 7:
					HOG(&newimg, &featureVector);
                    break;
			    /* Contour */
				case 8:
                    contourExtraction(&newimg, &featureVector);
                    break;
				case 9:
                    break;
				default: 
                    cout << "Error: this description method does not exists." << endl;
                    return -1;
			}

			labels.at<uchar>(imgTotal,0) = (uchar)i;
			for(k = 0; k < featureVectorSize; k++) {
				features.at<float>(imgTotal,k) = featureVector.at<float>(0, k);
			}
			imgTotal++;
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

	cout << "Wrote on file " << nome << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;
	for (i = 0; i < imgTotal; i++) {
	    // Write the image number and the referenced class
		fprintf(arq, "%d\t%d\t", i, labels.at<uchar>(i,0));
		for(k = 0; k < featureVectorSize; k++) {
			if (normalization == 2)  {
				fprintf(arq,"%.f ", features.at<float>(i, k));
			}
			else
				fprintf(arq,"%.5f ", features.at<float>(i, k));
		}
		fprintf(arq,"\n");	
	}

	// cout << ">>>>>>>> Wrote on data file " << nome << endl;
	// cout << "---------------------------------------------------------------------------------------" << endl;
 //    FILE *arqVis = fopen((nome+"data").c_str(), "w+");
	// int w, z;
	// fprintf(arqVis,"%s\n", "DY");
	// fprintf(arqVis,"%d\n", labels.size().height);
	// fprintf(arqVis,"%d\n", features.size().width);
	// for(z = 0; z < features.size().width-1; z++) {
	//     fprintf(arqVis,"%s%d;", "attr",z);
	// }
	// fprintf(arqVis,"%s%d\n", "attr",z);
	// for (w = 0; w < labels.size().height; w++) {
	//     fprintf(arqVis,"%d%s;", w,".jpg");
	//     for(z = 0; z < features.size().width; z++) {
	//         fprintf(arqVis,"%.5f;", features.at<float>(w, z));
	//     }
	//     float numeroimg =  labels.at<uchar>(w,0);
	//     fprintf(arqVis,"%1.1f\n", numeroimg);
	//     // cout << labels.at<float>(w,0) << " versus " << numeroimg << endl;
	// }

	if (method == 4) {
		for (i = 0; i < colors; i++) {
			free(coocurrenceMatrix[i]);
		}
		free(coocurrenceMatrix);
	}
	
	fclose(arq);
	return 0;
}

/**
 * 	Authors:
 *  	Luciana Calixta Escobar
 *		Gabriela Thumé
 *
 * 	Universidade de São Paulo / ICMC
 **/


#include "funcoesArquivo.h"

int qtdArquivos(char *directory){

 	int count = 0;
 	struct dirent *sDir = NULL;
 	DIR *dir = NULL;

 	dir = opendir(directory);
 	if(dir == NULL) {
 		fprintf(stderr,"Erro! There is no directory named %s\n", directory);
 		exit(-1);
 	}

	// conta quantos arquivos existem no directory
 	while(sDir = readdir(dir)) {
 		if( (strcmp(sDir->d_name, ".") != 0) &&
 			(strcmp(sDir->d_name, "..") != 0) &&
 			(strcmp(sDir->d_name, ".directory") != 0) &&
 			(strcmp(sDir->d_name, ".DS_Store") != 0) &&
 			(strcmp(sDir->d_name, ".directory") != 0)) {
 			count++;
 		}
 	}

 closedir(dir);

 return count;
}


int qtdImagensTotal(const char *base, int qtdClasses, int *objClass, int *maxs){

	int i, count = 0;
	char *directory = (char*)calloc(256, sizeof(char));
	*maxs = 0;

	for (i = 1; i <= qtdClasses; i++){
		sprintf(directory, "%s/%d/", base, i);
		int currentSize = qtdArquivos(directory);
		objClass[i-1] = currentSize;
		count += currentSize;
		if (currentSize > *maxs || *maxs == 0)
			*maxs = currentSize;
	}

	return count;
}

int descriptor(char const *baseImagem, char const *featuresDirectory, int method, int numberColor, double nRes, int oNorm, int *param, int nparam, int oZero, int quantMethod){

	int i, j, k, qtdImagem = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0;
	int featureVectorSize = 0, resizingFactor = (int)(nRes*100), maxc = 0;
	float min, max, normFactor;
	char *directory = (char*)calloc(128, sizeof(char));
	char nome[256], nometeste[256];
	double **Cm = NULL;
	Mat img, featureVector, features, labels, newimg;
	FILE *arq;
	char *quantMethodsNames[4] = {(char *)"Intensity", (char *)"Gleam", (char *)"Luminance", (char *)"MSB"};

	cout << "\n---------------------------------------------------------------------------------------" << endl;
	cout << "Image feature extraction" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	switch (method) {
		case 1: 
		sprintf(nome,"%s/%s_BIC_%s_%dc_%dr.txt", featuresDirectory, baseImagem, quantMethodsNames[quantMethod-1], numberColor, resizingFactor);
		featureVectorSize = numberColor*2;
		cout << "BIC and " << quantMethodsNames[quantMethod-1] << ":";
		break;
		case 2:
		sprintf(nome,"%s/%s_GCH_%s_%dc_%dr.txt", featuresDirectory, baseImagem, quantMethodsNames[quantMethod-1], numberColor, resizingFactor);
		featureVectorSize = numberColor;
		cout << "GCH and " << quantMethodsNames[quantMethod-1] << ":";
		break;
		case 3:
		sprintf(nome,"%s/%s_CCV_%s_%dc_%dr.txt", featuresDirectory, baseImagem, quantMethodsNames[quantMethod-1],  numberColor, resizingFactor);
		featureVectorSize = numberColor*2;
		cout << "CCV and " << quantMethodsNames[quantMethod-1] << ":";
		break;
		case 4:
		sprintf(nome,"%s/%s_Haralick6_%s_%dc_%dr.txt", featuresDirectory, baseImagem, quantMethodsNames[quantMethod-1],  numberColor, resizingFactor);
		Cm = (double **)calloc(numberColor, sizeof(double*));
		for (i=0; i<numberColor; i++) {
			Cm[i]= (double *)calloc(numberColor, sizeof(double));
		}
		featureVectorSize = 6;
		cout << "Haralick-6 and " << quantMethodsNames[quantMethod-1] << ":";
		break;
		case 5:
		sprintf(nome,"%s/%s_ACC_%s_%dc_%dd_%dr.txt",  featuresDirectory, baseImagem, quantMethodsNames[quantMethod-1], numberColor, nparam, resizingFactor);
		featureVectorSize = (numberColor*nparam);
		cout << "ACC and " << quantMethodsNames[quantMethod-1] << " c/ " << nparam << " distancias : ";
		break;
		default:
		break;
	}
	
	featureVector.create(1, featureVectorSize, CV_64F); // aloca o vetor de caracteristicas com o tamanho featureVectorSize
	featureVector = Scalar::all(0); // preenche com zeros
	
	cout << numberColor << " cores, tamanho " << nRes << endl;
	cout << "File: " << nome << endl;
	
	arq = fopen(nome, "w+");
	
	sprintf(directory, "%s/", baseImagem);
	
	// pega a quantidade de classes (i.e. verifica quantas subpastas existem)
	qtdClasses = qtdArquivos(directory); 
	
	//file .txt grava, na primeira linha, a quantidade de imagens,quantidade de classes, 
	//quantidade de atributos
	int *objperClass = (int *)malloc(qtdClasses*sizeof(int));
	
	qtdImgTotal = qtdImagensTotal(baseImagem, qtdClasses, objperClass, &maxc);
	
	fprintf(arq,"%d\t%d\t%d\n", qtdImgTotal, qtdClasses, featureVectorSize); 
	cout << "Objects: " << qtdImgTotal << " - Classes: " << qtdClasses << " - Features: " << featureVectorSize << endl;
	for (i = 0; i < qtdClasses; i++) {
		int bars = (int)(((float)objperClass[i]/(float)qtdImgTotal)*50.0);
		cout << (i+1) << " ";
		for (j = 0; j < bars; j++){
			cout << "|";
		}	  
		float porc = (float)objperClass[i]/(float)qtdImgTotal;
		cout << " " << porc*100 << "%" << " (" << objperClass[i] << ")" <<endl;
	}

	// pos processa matriz de dados para retirar colunas nulas
	// para isso constroi matriz para armazenar vetores
 	//if (oZero == 1) {
	features = Mat::zeros(qtdImgTotal, featureVectorSize, CV_32F);
	labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
 	//}
	
	for(i = 1; i <= qtdClasses; i++) {

		sprintf(directory,"%s/%d/", baseImagem, i); 
		qtdImagem = qtdArquivos(directory);
		cout << "classe " << i << " : " << directory << "\n";
		
		for(j = 0; j < qtdImagem; j++)	{
			
			sprintf(directory,"%s/%d/%d.jpg", baseImagem, i, j); 
			img = imread(directory, CV_LOAD_IMAGE_COLOR);
			
			if (img.empty()) { 
				fprintf(stderr,"Erro ao abrir imagem %s\n", directory); 
				return -1; 
			}
			
			if (nRes < 1) {
				resize(img, newimg, Size(), nRes, nRes, INTER_AREA);
				if (imgTotal==0) 
					imwrite("out.jpg", newimg);
			} else
			img.copyTo(newimg);
			
			//Quantizacao ou conversao de cores
			switch(quantMethod){
				case 1:
				QuantizationIntensity(newimg, newimg, numberColor);
				break;
				case 2:
				QuantizationGleam(newimg, newimg, numberColor);
				break;
				case 3:
				QuantizationLuminance(newimg, newimg, numberColor);
				break;
				case 4:
				QuantizationMSB(newimg, newimg, numberColor);
				break;
				default: 
				cout << "ERRO: metodo de quantizacao nao existe!!!" << endl;
				exit(-1);
			}
			
			switch(method){
	            /* BIC: image, descriptor, number of colors, normalization */
				case 1:
				BIC(newimg, featureVector, numberColor, oNorm);
				break;
			    /* GCH: image, descriptor, number of colors, normalization */
				case 2:
				GCH(newimg, featureVector, numberColor, oNorm);
				break;
			    /* CCV: image, descriptor, number of colors, normalization, 
			          limiar coerente/incoerente */
				case 3:
				CCV(newimg, featureVector, numberColor, oNorm, param[0]);
				break;
			    /* HARALICK: image, co-occurrence matriz, descriptor
			               numero de cores, normalization */
				case 4:
				HARALICK(newimg, Cm, featureVector, numberColor, oNorm);
				break;
			    /* ACC: image, descriptor, number of colors, normalization,
			          distance vector, distance number */
				case 5:
				ACC(newimg, featureVector, numberColor, oNorm, param, nparam);
				break;
			}

			labels.at<uchar>(imgTotal,0) = (uchar)i;
			
			for(k = 0; k < (featureVectorSize); k++) {
				features.at<float>(imgTotal,k) = featureVector.at<float>(0, k);
			}
			
			imgTotal++;
		}
	}
	
	if(method == 4 && oNorm != 0){
		normFactor = (oNorm == 1) ? 1.0 : 255.0;

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
	    // grava o numero da imagem e a classe que ela pertence
		fprintf(arq, "%d\t%d\t", i, labels.at<uchar>(i,0));
		for(k = 0; k < featureVectorSize; k++) {
			if (oNorm == 2)  {
				fprintf(arq,"%.f ", features.at<float>(i, k));
			}
			else
				fprintf(arq,"%.5f ", features.at<float>(i, k));
		}
		fprintf(arq,"\n");	
	}
	
	if (method == 4) {
		for (i=0; i<numberColor; i++) {
			free(Cm[i]);
		}
		free(Cm);
	}
	
	fclose(arq);
	return 1;
}

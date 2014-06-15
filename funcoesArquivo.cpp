/**
 * 
 *		Luciana Calixta Escobar
 *
 *
 *		A compilacao desse arquivo deve ser realizada por:
 *
 *
 **/


#include "funcoesArquivo.h"


int qtdArquivos(char *diretorio)
{
	struct dirent *sDir = NULL;
	DIR *dir = NULL;
	int contador = 0;
	
	// abre o diretorio
	dir = opendir(diretorio);
	
	// se der erro, retorna -1
	if(dir == NULL)
	{
		fprintf(stderr,"Erro!! Nao foi possivel abrir o diretorio %s!!!\n", diretorio);
		return -1;
	}
	// conta quantos arquivos existem no diretorio
	while(sDir = readdir(dir))
	{
		if( (strcmp(sDir->d_name, ".") != 0) &&
			(strcmp(sDir->d_name, "..") != 0) &&
			(strcmp(sDir->d_name, ".directory") != 0) )
		{
			contador++; 
// 			printf("%s %d\n", sDir->d_name, contador);
		}
	}
	
	closedir(dir);
	
	return contador;
}


int qtdImagensTotal(const char *base, int qtdClasses, int *objClass, int *maxs)
{
	int i;
	int contador = 0;
	char *diretorio = (char*)calloc(256, sizeof(char));
	*maxs = 0;
	
	for (i = 1; i <= qtdClasses; i++) 
	  {	  
		sprintf(diretorio, "%s/%d/", base, i);
	
		int currentsize = qtdArquivos(diretorio);
		objClass[i-1] = currentsize;
		contador += currentsize;
		if (currentsize > *maxs || *maxs == 0) *maxs = currentsize;
	  }
	
	  return contador;
}

int descriptor(const char *baseImagem, int method, int nColor, double nRes, int oNorm, int *param, int nparam, int oZero, int quantMethod)
{
	int qtdImagem = 0;
	int qtdClasses = 0;
	int qtdImgTotal = 0;
	char *diretorio = (char*)calloc(128, sizeof(char));
	int i, j, k;
	char nome[256];
	char nometeste[256];
	Mat img;
	FILE *arq;
	
	Mat mdesc; // vetor de caracteristicas de cada imagem
	
	double **Cm = NULL; // matriz de coocorrencia (se for usar Haralick)
	
	int rfactor = (int)(nRes*100); // fator de redimensionamento
	
	int fvsize = 0; // tamanho do vetor de caracteristicas
	
	char *quantMethodsNames[4] = {"Intensity", "Gleam", "Luminance", "MSB"};
	
	if (method == 1) {
	    sprintf(nome,"%s_BIC_%s_%dc_%dr.txt",baseImagem, quantMethodsNames[quantMethod-1], nColor, rfactor);
	    fvsize = nColor*2; // bic eh numero de cores * 2
	    cout << "BIC and " << quantMethodsNames[quantMethod-1] << ":";
	}
	else if (method == 2) {
	    sprintf(nome,"%s_GCH_%s_%dc_%dr.txt",baseImagem, quantMethodsNames[quantMethod-1], nColor, rfactor);
	    fvsize = nColor; // gch eh = numero de cores
	    cout << "GCH and " << quantMethodsNames[quantMethod-1] << ":";
	}
	else if (method == 3) {
	    sprintf(nome,"%s_CCV_%s_%dc_%dr.txt", baseImagem, quantMethodsNames[quantMethod-1],  nColor, rfactor);
	    fvsize = nColor*2;
	    cout << "CCV and " << quantMethodsNames[quantMethod-1] << ":";
	}
	else if (method == 4) {
	    sprintf(nome,"%s_Haralick6_%s_%dc_%dr.txt", baseImagem, quantMethodsNames[quantMethod-1],  nColor, rfactor);
	    Cm = (double **)calloc(nColor, sizeof(double*));
	    for (i=0; i<nColor; i++) {
		    Cm[i]= (double *)calloc(nColor, sizeof(double));
	    }
	    fvsize = 6;
	    cout << "Haralick-6 and " << quantMethodsNames[quantMethod-1] << ":";
	}
	else if (method == 5) {
	    sprintf(nome,"%s_ACC_%s_%dc_%dd_%dr.txt",baseImagem, quantMethodsNames[quantMethod-1], nColor, nparam, rfactor);
	    fvsize = (nColor*nparam);
	    cout << "ACC and " << quantMethodsNames[quantMethod-1] << " c/ " << nparam << " distancias : ";
	}
	
	mdesc.create(1, fvsize, CV_64F); // aloca o vetor de caracteristicas com o tamanho fvsize
	mdesc = Scalar::all(0); // preenche com zeros
	
	cout << nColor << " cores, tamanho " << nRes << endl;
	cout << "Arquivo: " << nome << endl;
	
	arq = fopen(nome, "w+");
	
	sprintf(diretorio, "%s/", baseImagem);
	
	// pega a quantidade de classes (i.e. verifica quantas subpastas existem)
	qtdClasses = qtdArquivos(diretorio); 
	
	//arquivo .txt grava, na primeira linha, a quantidade de imagens,quantidade de classes, 
	//quantidade de atributos
	int *objperClass = (int *)malloc(qtdClasses*sizeof(int));
	
	int maxc = 0;
	qtdImgTotal = qtdImagensTotal(baseImagem, qtdClasses, objperClass, &maxc);
	
	fprintf(arq,"%d\t%d\t%d\n", qtdImgTotal, qtdClasses, fvsize); 
	cout << "Objects: " << qtdImgTotal << " - Classes: " << qtdClasses << " - Features: " << fvsize << endl;
	for (i = 0; i < qtdClasses; i++) {
	    int bars = (int)(((float)objperClass[i]/(float)qtdImgTotal)*50.0);
	    cout << (i+1) << " ";
	    for (j = 0; j < bars; j++) {
		cout << "|";
	    }	  
	    float porc = (float)objperClass[i]/(float)qtdImgTotal;
	    cout << " " << porc*100 << "%" << " (" << objperClass[i] << ")" <<endl;
	}
	
	int imtotal = 0;
	Mat features, labels;
	
	// pos processa matriz de dados para retirar colunas nulas
	// para isso constroi matriz para armazenar vetores
 	//if (oZero == 1) {
	features = Mat::zeros(qtdImgTotal, fvsize, CV_32F);
 	labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
 	//}
	
	Mat newimg;
	for(i = 1; i <= qtdClasses; i++) {
	  
		// pega a quantidade de imagens de cada classe
		sprintf(diretorio,"%s/%d/", baseImagem, i); 
		qtdImagem = qtdArquivos(diretorio);

		cout << "classe " << i << " : " << diretorio << "\n";
		
		// para cada imagem
		for(j = 0; j < qtdImagem; j++)	{
			
			sprintf(diretorio,"%s/%d/%d.jpg", baseImagem, i, j); 
			img = imread(diretorio, CV_LOAD_IMAGE_COLOR);
			
			if (img.empty()) { fprintf(stderr,"Erro ao abrir imagem %s\n", diretorio); return -1; }
			
			if (nRes < 1) {
			  // compute new height and widt
			  resize(img, newimg, Size(), nRes, nRes, INTER_AREA);
			  if (imtotal==0) imwrite("out.jpg", newimg);
			} else {
			  img.copyTo(newimg);
			}
			
			//Quantizacao ou conversao de cores
			switch(quantMethod){
			  case 1:
			    QuantizationIntensity(newimg, newimg, nColor);
			    break;
			  case 2:
			    QuantizationGleam(newimg, newimg, nColor);
			    break;
			  case 3:
			    QuantizationLuminance(newimg, newimg, nColor);
			    break;
			  case 4:
			    QuantizationMSB(newimg, newimg, nColor);
			    break;
			  default: 
			    cout << "ERRO: metodo de quantizacao nao existe!!!" << endl;
			    exit(-1);
			}
			
			// different descriptors
			if (method == 1) {
			    // BIC: imagem, descritor, numero de cores, normalizacao
			    BIC(newimg, mdesc, nColor, oNorm);
			}
			else if (method == 2) {
			    // GCH: imagem, descritor, numero de cores, normalizacao
			    GCH(newimg, mdesc, nColor, oNorm);
			}
			else if (method == 3) {
			    // CCV: imagem, descritor, numero de cores, normalizacao, 
			    //      limiar coerente/incoerente
			    CCV(newimg, mdesc, nColor, oNorm, param[0]);
			}
			else if (method == 4) {
			    // HARALICK: imagem, matriz de co-ocorrencia, descritor
			    //           numero de cores, normalizacao
			    HARALICK(newimg, Cm, mdesc, nColor, oNorm);
			}
			else if (method == 5) {
			    // ACC: imagem, descritor, numero de cores, normalizacao,
			    //      vetor de distancias, numero de distancias
			    ACC(newimg, mdesc, nColor, oNorm, param, nparam);
			}
		
			labels.at<uchar>(imtotal,0) = (uchar)i;
			
			for(k = 0; k < (fvsize); k++) {
				features.at<float>(imtotal,k) = mdesc.at<float>(0, k);
			}
			
			imtotal++;
		}
	}
	
	if(method == 4 && oNorm != 0){
  	float min, max;
	  float normFactor = (oNorm == 1) ? 1.0 : 255.0;

	  for(int j = 0; j < features.cols; ++j){
	    min = features.at<float>(0,j); 
	    max = features.at<float>(0,j);
	    
	    for(int i = 1; i < features.rows; ++i){
	      if(features.at<float>(i,j) > max){
	        max = features.at<float>(i,j);
	      }
	      
	      if (features.at<float>(i,j) < min){
	        min = features.at<float>(i,j);
	      }
	    }
	    
	    for(int i = 0; i < features.rows; ++i){
	      features.at<float>(i,j) = normFactor * ((features.at<float>(i,j) - min) / (max - min));
	    }
	  }
	}

	cout << "Grava no arquivo " << nome << " com : " << imtotal << " imagens" << endl;
	for (i = 0; i < imtotal; i++) {
	    // grava o numero da imagem e a classe que ela pertence
	    cout << labels.at<uchar>(i,0) << endl;
	    fprintf(arq, "%d\t%d\t", i, labels.at<uchar>(i,0));
	    for(k = 0; k < (fvsize); k++) {
		  if (oNorm == 2)  {
		   //   features.at<double>(
		      fprintf(arq,"%.f ", features.at<float>(i, k));
		  }
		  else
		      fprintf(arq,"%.5f ", features.at<float>(i, k));
	    }
   	    fprintf(arq,"\n");	
	}
	
	if (method == 4) {
	    for (i=0; i<nColor; i++) {
		    free(Cm[i]);
	    }
	    free(Cm);
	}
	
	fclose(arq);
	return 1;
}

/**
 * 
 *		Luciana Calixta Escobar
 *
 *
 *		A compilacao desse arquivo deve ser realizada por:
 *		@g++ -c funcoesAux.c -I /usr/include/opencv -lcv -lml -lhighgui -lcvauc
 *
 *
 **/


#include "funcoesAux.h"


/*	Funcao Intensidade
 Conversão de uma imagem colorida para a escala de cinza usando os canais lineares
 Requer:
	- imagem a ser convertida
	- imagem onde sera armazenada a imagem convertida
*/
void QuantizationIntensity(Mat &I, Mat &Q, int nColors) 
{
    //cout << " >>> Intensity\n";
    vector<Mat> imColors(3);

    // a partir da imagem 'out' (ja convertida), separa os canais nas
    // tres matrizes definidas em imColors
    split(I, imColors);

    //somatorio de 1/3 de cada canal em cada pixel
    Q = (imColors[0]/3) + (imColors[1]/3) + (imColors[2]/3);
    
   if (nColors < 256) {
      double min, max;
      Point maxLoc, minLoc;
      minMaxLoc(Q, &min, &max, &minLoc, &maxLoc);
      double stretch = ((double)((nColors-1)) / (max - min ));

      Q = Q - min;
      Q = Q * stretch;

      /*// se nao funcionar, use a funcao com loop
      int x, y;
      for (x = 0; x < imSize.height; x++) {
         for (y = 0; y < imSize.width; y++) {
             IMAGEM.at<uchar>(x,y) = (uchar) ((double)IMAGEM.at<uchar>(x,y) - (min)) * stretch;
         }
      }*/  
    }  
    
//     imshow("convertida", Q);                   // mostra imagem reduzida
//     waitKey(0);
}

/*	Funcao Gleam
 Conversão de uma imagem colorida para a escala de cinza usando os canais com a correção gamma
 Requer:
	- imagem a ser convertida
	- imagem onde sera armazenada a imagem convertida
*/

void QuantizationGleam(Mat &I, Mat &Q, int nColors) 
{
    //cout << " >>> Gleam\n";
    vector<Mat> imColors(3);
    //potencia usada para a correção gamma dos canais
    double pot = 1/2.2;

    // a partir da imagem 'out' (ja convertida), separa os canais nas
    // tres matrizes definidas em imColors
    split(I, imColors);

    Mat res0_32;
    Mat res1_32;
    Mat res2_32;
    
    imColors[0].convertTo(res0_32, CV_32F);
    imColors[1].convertTo(res1_32, CV_32F);
    imColors[2].convertTo(res2_32, CV_32F);
    
    pow(res0_32,pot,res0_32);
    pow(res1_32,pot,res1_32);
    pow(res2_32,pot,res2_32);
    
    res0_32.convertTo(imColors[0], CV_8U);
    res1_32.convertTo(imColors[1], CV_8U);
    res2_32.convertTo(imColors[2], CV_8U);
    
    //res2_32.convertTo(imColors[2], CV_8U);
    
    //somatorio de 1/3 de cada canal com a correção gamma em cada pixel
    Q = imColors[0]/(3.0) + imColors[1]/(3.0) + imColors[2]/(3.0);

    //Q = res0_32/(3.0) + res1_32/(3.0) + res2_32/(3.0);
    
    if (nColors < 256) {
      double min, max;
      Point maxLoc, minLoc;
      minMaxLoc(Q, &min, &max, &minLoc, &maxLoc);
      double stretch = ((double)((nColors-1)) / (max - min ));

      Q = Q - min;
      Q = Q * stretch;

      /*// se nao funcionar, use a funcao com loop
      int x, y;
      for (x = 0; x < imSize.height; x++) {
         for (y = 0; y < imSize.width; y++) {
             IMAGEM.at<uchar>(x,y) = (uchar) ((double)IMAGEM.at<uchar>(x,y) - (min)) * stretch;
         }
      }*/  
    }
    
//     imshow("convertida", Q);                   // mostra imagem reduzida
//     waitKey(0);
}


/*	Funcao Luminance
 Conversão de uma imagem colorida para a escala de cinza usando uma combinação com pesos para cada canal
 Requer:
	- imagem a ser convertida
	- imagem onde sera armazenada a imagem convertida
*/
void QuantizationLuminance(Mat &I, Mat &Q, int nColors) 
{
    //cout << " >>> Luminance\n";
    vector<Mat> imColors(3);

    // a partir da imagem 'out' (ja convertida), separa os canais nas
    // tres matrizes definidas em imColors
    split(I, imColors);

    //somatorio de cada canal com seu respectivo peso em cada pixel
    Q = (0.3*imColors[0]) + (0.59*imColors[1]) + (0.11*imColors[2]);
    
    if (nColors < 256) {
      double min, max;
      Point maxLoc, minLoc;
      minMaxLoc(Q, &min, &max, &minLoc, &maxLoc);
      double stretch = ((double)((nColors-1)) / (max - min ));

      Q = Q - min;
      Q = Q * stretch;

      /*// se nao funcionar, use a funcao com loop
      int x, y;
      for (x = 0; x < imSize.height; x++) {
         for (y = 0; y < imSize.width; y++) {
             IMAGEM.at<uchar>(x,y) = (uchar) ((double)IMAGEM.at<uchar>(x,y) - (min)) * stretch;
         }
      }*/  
    }
    
//     imshow("convertida", Q);                   // mostra imagem reduzida
//     waitKey(0);
}


/*	Funcao QuantizationMSB
 Quantiza uma imagem de acordo com a quantidade de cores passada por argumento
 Requer:
	- imagem a ser quantizada
	- imagem onde sera armazenada a imagem quantizada
	- quantidade de cores 
 Retorna: 
	- quantidade unicas de cores da imagem quantizada */
int QuantizationMSB(Mat &I, Mat &Q, int nColors) 
{
	// pega tamanho da imagem
	Size imgSize = I.size();
	
	int bitsc = log(nColors)/log(2); // calcula numero de bits necessarios
	int cc = (int)(bitsc/3); // calcula numero de bits por canal
	int RGBb[3]={cc,cc,cc}; // atribui os bits por canal
	int rest = (bitsc % 3); // verifica se ha sobra de bits, armazena o resto
	int k;
	// atribui os bits restantes aos canais R, G e B, em sequencia
	for (k = 0 ;rest > 0; rest--, k = (k+1)%3) {
		RGBb[k]++;
	}
	
	// vetor para armazenar a frequencia de cada cor 
	int freq[nColors];
	for (int i = 0; i < nColors; i++) {
		freq[i] = 0;
	}
	int unique = 0;
	
	MatIterator_<Vec3b> it, end;
	MatIterator_<uchar> it2, end2;
	for( it2 = Q.begin<uchar>(), end2 = Q.end<uchar>(), it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it, ++it2)
	{
		uchar dR = (8-RGBb[0]);
		uchar dG = (8-RGBb[1]);
		uchar dB = (8-RGBb[2]);
		// mascara para realizar AND em cada canal
		uchar Ra = ((int)(pow(2,RGBb[0]))-1) << dR;
		uchar Ga = ((int)(pow(2,RGBb[1]))-1) << dG;
		uchar Ba = ((int)(pow(2,RGBb[2]))-1) << dB;
		
		uchar R = (*it)[0];
		uchar G = (*it)[1];
		uchar B = (*it)[2];
		
		// operacao para obter MSBs em cada canal
		uchar C1 = (R & Ra) >> dR;                  // extrai MSBs de R e move p/ o final 
		uchar C2 = (G & Ga) >> (dR-RGBb[1]);        // extrai MSBs de G e move p/ apos C1
		uchar C3 = (B & Ba) >> (dR-RGBb[1]-RGBb[2]);// extrai MSBs de B e move p/ apos C2
		
		uchar newcolor = C1 | C2 | C3; // operador | equivalente a 'OU', faz a fusao dos tres componentes
		
		(*it2) = newcolor;
		
		if (newcolor > 255) cout << "color overflow: " << newcolor << endl;
		
		// conta quantas cores unicas foram obtidas
		if (freq[newcolor] == 0) unique++;
		
		// frequencia de cada cor quantizada
		freq[newcolor]++;
	}	
	
	return unique;
}

void RemoveNullColumns(Mat &Feat) {
      int height = Feat.size().height;
      int width = Feat.size().width;
      
      vector<int> marktoremove(width);
      for (int i = 0; i < width; i++) {
	  Mat columni = Feat.col(i);
	  double sumi = sum(columni)[0];
	  if (sumi == 0) marktoremove[i] = 1;
	  cout << "Sum " << i << " : " << sumi << endl;
      }
}

/* Normaliza um histograma
 * Funcao para normalizar (entre 0 e 255) o histograma gerado pelo descritor BIC
 * Requer:
 *	- o histograma a ser normalizado
 *	- um histograma ja alocado, para guardar o resultado
 *	- o tamanho do vetor
 *	- fator de normalizacao */
void NormalizeHist(long int *hist, float *histnorm, int nColor, int fator) 
{
	int i;
	long int sum = 0;
	long int max = hist[0];
	float e = 0.01;
	
	
	for (i = 0; i < nColor ; i++)
	{
		//sum += hist[i]*hist[i];
		sum += hist[i];
		max = (hist[i] > max) ? hist[i] : max;
	}
	
	if (fator == 1)
	{
		for (i = 0; i < nColor ; i++) 
		{
			histnorm[i] = hist[i]/((float)sum+e);
		}
	} 
	else if (fator > 1) 
	{
		for (i = 0; i < nColor ; i++) 
		{
			histnorm[i] = (hist[i]/(float)max)*(float)fator;
		}
	}
}


/*	Funcao distManhattan
 * Calcula a diferenca entre dois histrogramas atraves da distancia Manhattan
 * Requer:
 *	- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
 *	- tamanho do histograma 
 * Retorna:
 *	- a distancia entre os dois histogramas */
double distManhattan(double *p, double *q, int size) 
{
	int i;
	double dist = 0;
	
	for (i = 0; i < size; i++) 
	{
		dist += fabs(p[i]-q[i]);      
	}
	
	return dist;  
}


/*	Funcao distEuclid
 * Calcula a diferenca entre dois histrogramas atraves da distancia Euclidiana
 * Requer:
 *	- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
 *	- tamanho do histograma 
 * Retorna:
 *	- a distancia entre os dois histogramas */
double distEuclid(double *q, double *p, int size)
{
	int i;
	double dist = 0;
	
	for(i = 0; i < size; i ++) 
	{
		dist = dist + pow((q[i]-p[i]),2);
	}
	
	dist = sqrt(dist);
	
	return dist;
}


/*	Funcao distChessboard
 * Calcula a diferenca entre dois histrogramas atraves da distancia Chessboard
 * Requer:
 *	- dois histogramas, com o mesmo tamanho, ja preenchidos com valores
 *	- tamanho do histograma 
 * Retorna:
 *	- a distancia entre os dois histogramas */
double distChessboard(double *p, double *q, int size)
{
	int i;
	double dist = 0;
	double maxVal = -1;
	
	for (i = 0; i < size; i++)
	{
		dist = fabs(p[i]-q[i]);
		if (maxVal < dist) { maxVal = dist; }
	}
	
	return maxVal;
}

/**
 * 
 *		Luciana Calixta Escobar
 *
 *
 *		A compilacao desse arquivo deve ser realizada por:
 *		@gcc -c descritores.c -I /usr/include/opencv -lcv -lml -lhighgui -lcvauc
 *
 *
 **/


#include "descritores.h"
#include "funcoesAux.h"


/* Funcao Find Neighbor
 * Encontra os vizinhos de um pixel
 * Usada no descritor CCV
 * Requer:
 *	- imagem original
 *	- fila de pixels
 *	- pixels ja visitados
 *	- tamanho da regiao */
void find_neighbor(Mat & img, queue<Pixel> & pixels, int * visited, long int & tam_reg)
{
	// testa os quatro vizinhos
	// se podem ser visitados então marca visited, adiciona na fila e incrementa tam_reg
	int height = img.rows;
	int width = img.cols;

	// pega o pixel da frente da fila
	Pixel pix = pixels.front();
	// retira o pixel da fila
	pixels.pop();

	// pega informacoes do pixel
	int i = pix.i;
	int j = pix.j;
	uchar pix_color = pix.color;
	uchar img_color;
	
	int s, t; // indices dos vizinhos
	
	// testa os quatro vizinhos
	// verifica se esta na borda e se a cor eh a mesma
	s = i - 1;
	t = j;
	
	
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix); // acima
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}


	s = i;
	t = j + 1;
	
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix); // a direita
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}


	s = i + 1;
	t = j;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix); // abaixo
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}

	s = i;
	t = j - 1;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix); // a esquerda
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}
	
}


/* Descritor CCV
 * Cria dois histogramas de cor da imagem:
 * 1 -> histograma de pixels coerentes
 * 2 -> histograma de pixels incoerentes
 * Requer:
 *	- imagem original
 *	- fistograma ja alocado, com duas vezes a quantidade de cor
 *	- quantidade de cores usadas na imagem
 *	- nivel de coerencia */	
void CCV(Mat & img, Mat &features, int nColor, int oNorm, int threshold)
{

	int i, j;
	//float *descriptor = new float[nColor*2];
	long int *descriptor = new long int[nColor*2];

	for(i = 0; i < nColor*2; i++)
	{
		descriptor[i] = 0;
	}

	Mat img_quant(img.size(), CV_8UC1);
	int height   = img_quant.rows;       // altura da imagem (eixo x)
	int width    = img_quant.cols;        // largura da imagem (eixo y)

	if (img.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      img_quant = img - min;
	      img_quant = img_quant * stretch;
	}
	  else {
	      QuantizationMSB(img, img_quant, nColor);
	}

	// array to mark
	int *pxl_visited = new int[height*width]();
	long int tam_reg;

	// fila para processar pixels
	queue<Pixel> pixels;
	// cada pixel a ser processado
	Pixel pix;


	// for each pixel, visit neighbors in order to find the regions
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			// if the current pixel was NOT already visited
			if (!pxl_visited[i*width + j])
			{
				// STORE current pixel values
				pix.i = i; 
				pix.j = j;
				pix.color = img_quant.at<uchar>(i, j);

				// put pixel on the queue
				pixels.push (pix);
				// mark as visited
				pxl_visited[i*width + j] = 1;
				
				// start new region, with size = 1
				tam_reg = 1;
		
				// while queue is NOT empty, find neighbors
				while (!pixels.empty())
					find_neighbor(img_quant, pixels, pxl_visited, tam_reg);

				// check if size of the region is higher than the coherence threshold
				if(tam_reg >= threshold)
					descriptor[pix.color*2 + 0] += tam_reg; // coerente
				  else
					descriptor[pix.color*2 + 1] += tam_reg; // incoerente
			}
		}
	}

	if (oNorm == 0) {
	  for (i = 0; i < nColor*2 ; i++) 
	  {
		features.at<float>(0,i) = (float)descriptor[i];
	  }
	}
	else {
	    float *norm = new float[nColor*2];
	    if (oNorm == 1) 
		NormalizeHist(descriptor, norm, 2*nColor, 1);
	    else if (oNorm == 2) 
		NormalizeHist(descriptor, norm, 2*nColor, 255);
	    for (i = 0; i < nColor*2 ; i++) 
	    {
		 features.at<float>(0,i) = norm[i];// copia no vetor "features"
	    }
	    delete[] norm;
	}
	
	delete[] pxl_visited;
	delete[] descriptor;

}


/* Descritor GCH
 * Cria histrograma de cor da imagem.
 * Requer:
 *	- imagem original
 *	- histrograma ja alocado
 *	- quantidade de cores usadas na imagem */
void GCH(Mat &I, Mat &features, int nColor, int oNorm)
{
	// aloca uma nova imagem do tamanho da original
	Mat Q(I.size(), CV_8U, 1);
	
	if (I.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	  else {
	      QuantizationMSB(I, Q, nColor);
	}
	
	int i, j;
	long int *hist = new long int[nColor];
	
	for (i = 0; i < nColor; i++) 
	{
		hist[i] = 0;    // Initialize all elements to zero.
	}
	
	MatIterator_<uchar> it, end;
	end = Q.end<uchar>();
	
	for (it = Q.begin<uchar>(); it != end; ++it) 
	{
		hist[(*it)]++;  // incrementa o valor da cor encontrada no histograma
	}
	
	if (oNorm == 0) {
	  for (i = 0; i < nColor ; i++) 
	  {
		features.at<float>(0,i) = (float)hist[i];
	  }
	}
	else {
	    float *norm = new float[nColor];
	    if (oNorm == 1) 
		NormalizeHist(hist, norm, nColor, 1);
	    else if (oNorm == 2) 
		NormalizeHist(hist, norm, nColor, 255);
	    
	    for (i = 0; i < nColor ; i++) 
	    {
		 features.at<float>(0,i) = norm[i];// copia no vetor "features"
	    }
	    delete[] norm;
	}
	
	delete[] hist;
}


/* Descritor BIC
 * Cria dois histrogramas de cor da imagem:
 * 1 -> histograma de borda
 * 2 -> histograma de interior
 * Requer:
 *	- imagem original
 *	- histograma ja alocado, com tamanho de duas vezes a quantidade de cor
 *	- quantidade de cores usadas na imagem 
 * No histograma, de 0 até (nColor -1) = Borda, de nColor até (2*nColor -1) = Interior */
void BIC(Mat &I, Mat &features, int nColor, int oNorm) 
{
	Size imgSize = I.size();
	int height = imgSize.height;
	int width = imgSize.width;
	
	Mat Q(imgSize, CV_8U, 1);
	
	if (I.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	  else {
	      QuantizationMSB(I, Q, nColor);
	}
	
	int i, j;
	long int *hist = new long int[2*nColor];

	for (i = 0; i < 2*nColor; i++) 
	{
		hist[i] = 0;    // Initialize all elements to zero.
	}
	
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++) 
		{
			uchar aux = Q.at<uchar>(i,j);
			if (i > 0 && j > 0 && j < width-1 && i < height-1) 
			{
				if ((Q.at<uchar>(i,j-1) == aux) && 
					(Q.at<uchar>(i,j+1) == aux) && 
					(Q.at<uchar>(i-1,j) == aux) && 
					(Q.at<uchar>(i+1,j) == aux)) 
				{
					hist[aux]++;
				}
				  else 
				  {
					hist[aux+nColor]++;
				  }
			}
			  else
			  {
				hist[aux+nColor]++;
			  }
		}
	}
	
	if (oNorm == 0) {
	  for (i = 0; i < nColor*2 ; i++) 
	  {
		features.at<float>(0,i) = (float)hist[i];
	  }
	}
	else {
	    float *norm = new float[nColor*2];
	    if (oNorm == 1) 
		NormalizeHist(hist, norm, nColor*2, 1);
	    else if (oNorm == 2) 
		NormalizeHist(hist, norm, nColor*2, 1024);
	    
	    for (i = 0; i < nColor*2 ; i++) 
	    {
		 features.at<float>(0,i) = norm[i];// copia no vetor "features"
	    }
	    delete[] norm;
	}

	delete[] hist;
}


/* Funcao CoocurrenceMatrix
 * Cria uma matriz que contem a ocorrencia de cada cor em cada pixel da imagem
 * Requer:
 *	- imagem original
 *	- matriz ja alocada, no tamanho nColor x nColor
 *	- quantidade de cores usadas na imagem
 *	- coordenadas dX e dY, que podem ser 0 ou 1 */
void CoocurrenceMatrix(Mat &I, double **Cm, int nColor, int dX, int dY) 
{
	int i,j;
	
	Mat Q(I.size(), CV_8U, 1);
	if (I.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	  else {
	      QuantizationMSB(I, Q, nColor);
	}

	// noma imagem quantizada chamada novaQ
	// alocar uma nova imagem de tamanho maior para ser processada
	Size newSize(I.rows+((dX+1)*2), I.cols+((dY+1)*2)); // cria um objeto 'tamanho'
	Mat novaQ(newSize, CV_8U, 1); // cria a imagem
	
	// copia a imagem quantizada e replica os pixels das bordas
	copyMakeBorder(Q, novaQ, (dX+1)*2, (dX+1)*2, (dY+1)*2, (dY+1)*2, BORDER_REPLICATE);
	
	Size imgSize = novaQ.size();
	int height = imgSize.height;
	int width = imgSize.width;
	
	// percorre a imagem
	for (i = dX; i < height-dX; i++)
	{
		for (j = dY; j < width-dY; j++) 
		{
			int pref = novaQ.at<uchar>(i,j); //pixel referencia
			int pviz = novaQ.at<uchar>((i+dX),(j+dY)); //pixel vizinho
			Cm[pref][pviz]++;
		}
	}
	
	double sum = 0;
	// calcula a soma das ocorrencias
	for (i = 0; i < nColor; i++) 
	{
		for (j = 0; j < nColor; j++) 
		{	
			sum += Cm[i][j];
		}
	}
	
	// normaliza a matriz de forma que sum(Cm) = 1
	for (i = 0; i < nColor; i++) 
	{
		for (j = 0; j < nColor; j++) 
		{
			Cm[i][j] /= sum;
		}
	}
}


/* Funcao Haralick
 * Cria um histograma com 6 descritores de textura
 * Requer:
 *	- matriz de coocorrencia
 *	- quantidade de cores usadas na imagem
 *	- histograma ja alocado
 * Os descritores sao:
 *	- maxima Probabilidade
 *	- correlacao
 *	- contraste
 *	- energia (uniformidade)
 *	- homogeneidade 
 *	- entropia */
void Haralick6(double **Cm, int nColor, Mat &features) 
{
	
	int i,j;
	double m_r = 0.0;
	double m_c = 0.0;
	double s_r = 0.0;
	double s_c = 0.0;
	double *Pi = (double *) calloc(nColor, sizeof(double));
	double *Pj = (double *) calloc(nColor, sizeof(double));
	
	for (i = 0; i < nColor; i++) 
	{
		for (j = 0; j<nColor; j++) 
		{
			Pi[i] += Cm[i][j];
		}
		m_r += i*Pi[i];	
	}
	
	for (j = 0; j < nColor; j++)
	{
		for (i = 0; i<nColor; i++) 
		{
			Pj[j] += Cm[i][j];
		}
		m_c += j*Pj[j];	
	}
	
	for (i = 0; i < nColor; i++) 
	{
		s_r += ((i-m_r)*(i-m_r)) * Pi[i];
		s_c += ((i-m_c)*(i-m_c)) * Pj[i];
	}
	s_r = sqrt(s_r);
	s_c = sqrt(s_c);
	
	double maxp = 0.0;
	double corr = 0.0;
	double cont = 0.0;
	double unif = 0.0;
	double homo = 0.0;
	double entr = 0.0;
	double auxv = 0.0;
	
	for (i = 0; i < nColor; i++)
	{
		for (j = 0; j<nColor; j++) 
		{
			auxv = Cm[i][j];
			
			// Descritor 1 - Maxima Probabilidade
			if (maxp < auxv) 
			{
				maxp = auxv;
			}
			
			// Descritor 2 - correlacao
			if (s_r > 0 && s_c > 0) 
			{
				corr += ((i-m_r)*(j-m_c)*auxv) / (s_r*s_c);	    
			}
			
			// Descritor 3 - contraste
			cont += ( (i-j)*(i-j)*auxv );
			
			// Descritor 4 - energia (uniformidade)
			unif += (auxv*auxv);
			
			// Descritor 5 - homogeneidade 
			homo += (auxv) / (1 + abs(i-j));
			
			// Descritor 6 - entropia
			if (auxv != 0) 
			{
				//entr+= auxv*( log(auxv) / log(2) );
				entr+= auxv*( log2(auxv));
			}
		}
	}
	
	entr = -entr;
	
	features.at<float>(0, 0) = maxp;
	features.at<float>(0, 1) = corr;
	features.at<float>(0, 2) = cont;
	features.at<float>(0, 3) = unif;
	features.at<float>(0, 4) = homo;
	features.at<float>(0, 5) = entr;  
}


/* Descritor Haralick
 * Cria um histograma com 6 descritores de textura
 * Chama as funcoes Haralick e CoocurrenceMatrix
 * Requer:
 * 	- imagem original
 *	- matriz de coocorrencia
 *	- quantidade de cores usadas na imagem
 *	- histograma ja alocado
 * Os descritores sao:
 *	- maxima Probabilidade
 *	- correlacao
 *	- contraste
 *	- energia (uniformidade)
 *	- homogeneidade 
 *	- entropia */
void HARALICK(Mat &I, double **Cm, Mat &features, int nColor, int oNorm) 
{
	CoocurrenceMatrix(I, Cm, nColor, 2, 0);
	Haralick6(Cm, nColor, features);
}


/* Descritor Autocorrelograma
 * Cria um histograma de cor que descreve a distribuição 
 * global da correlação entre a localização espacial de cores
 * Requer:
 *	- imagem original
 *	- valor da distancia k entre os pixels
 *	- histograma ja alocado
 *	- quantidade de cores usadas na imagem */
void ACC(Mat &I, Mat &features, int nColor, int oNorm, int *k, int totalk)
{
	// aloca uma nova imagem do tamanho da original
	// com 8 bits por pixel e 1 canal de cor
	Mat Q(I.size(), CV_8U, 1);
	
	QuantizationMSB(I, Q, nColor);
	
	int i,j, x, y;
	vector<long int> desc(nColor*totalk);
	
	double descNorm = 0;//acumulador para normalizar o descritor
	
	Size imgSize = Q.size();
	int height = imgSize.height;// altura
	int width = imgSize.width;// largura
	
	// finds the maximum distance inside 'k'
	int maxdist = 0;
	for (int d = 0; d < totalk; d++) {
	    if (k[d] > maxdist) maxdist = k[d];
	}

	// for each distance
	for (int d = 0; d < totalk; d++) {
	    int cd = k[d]; // current distance
	    
	    for (i = cd; i < (height-cd); i++) {
 		for (j = cd; j < (width-cd); j++) {
		      // chessboard distance (4 'lines' of a square)
		      // top : x = (i-cd), y = varying between (j-cd) and (j+cd)
		      x = (i-cd);
		      for (y = (j-cd); y <= (j+cd); y++) {
		       	   if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
 				int pos = (int)Q.at<uchar>(i,j);
 				desc[pos+(d*totalk)]++;
 				descNorm++;
 			   }
		      }
		      // bottom : x = (i+cd), y = varying between (j-cd) and (j+cd)
		      x = (i+cd);
		      for (y = (j-cd); y <= (j+cd); y++) {
		       	   if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
 				int pos = (int)Q.at<uchar>(i,j);
 				desc[pos+(d*totalk)]++;
 				descNorm++;
 			   }
		      }
		      // left : x = varying between (i-cd) and (i+cd), y = (i-cd) 
		      y = (i-cd);
		      for (x = (i-cd); x <= (i+cd); x++) {
		       	   if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
 				int pos = (int)Q.at<uchar>(i,j);
 				desc[pos+(d*totalk)]++;
 				descNorm++;
 			   }
		      }
		      // right : x = varying between (i-cd) and (i+cd), y = (i+cd) 
		      y = (i+cd);
		      for (x = (i-cd); x <= (i+cd); x++) {
		       	   if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
 				int pos = (int)Q.at<uchar>(i,j);
 				desc[pos+(d*totalk)]++;
 				descNorm++;
 			   }
		      }
		}
	    }
	}


	vector<float> norm(nColor*totalk);
	float descsum = 0;
	for (i = 0; i < (nColor*totalk) ; i++) 
	{
		norm[i] = (float)(desc[i]/(float)descNorm);
		descsum += norm[i];
	}
// 	cout << "Feature vector : ";
	if (oNorm == 0) {
		for (i = 0; i < nColor*totalk ; i++) 
		{
		      features.at<float>(0,i) = (float)desc[i];
		}
	} else if (oNorm == 1) {
	  for (i = 0; i < nColor*totalk ; i++) 
	  {
		features.at<float>(0,i) = norm[i];
//  	        fprintf(stdout, "%.3f ", features.at<float>(0,i));
	  }
	}
	else {
	    for (i = 0; i < nColor*totalk; i++) 
	    {
		 features.at<float>(0,i) = norm[i]*255;// copia no vetor "features"
	    }
	}
//         cout << endl;
}

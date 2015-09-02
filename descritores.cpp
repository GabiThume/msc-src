/**
 * 	Authors:
 *  	Luciana Calixta Escobar
 *		Gabriela Thumé
 *
 * 	Universidade de São Paulo / ICMC
 *
 **/

#include "descritores.h"
#include "quantization.h"

/* Funcao Find Neighbor
 * Encontra os vizinhos de um pixel
 * Usada no descritor CCV
 * Requer:
 *	- imagem original
 *	- fila de pixels
 *	- pixels ja visitados
 *	- tamanho da regiao */
void find_neighbor(Mat img, queue<Pixel> *pixels, int *visited, long int *tam_reg) {
	// testa os quatro vizinhos
	// se podem ser visitados então marca visited, adiciona na fila e incrementa tam_reg
	int height = img.rows;
	int width = img.cols;

	// pega o pixel da frente da fila
	Pixel pix = (*pixels).front();
	// retira o pixel da fila
	(*pixels).pop();

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
			(*pixels).push (pix); // acima
			visited[s*width + t] = 1;
			(*tam_reg)++;
	    }
	}

	s = i;
	t = j + 1;

	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
			pix.i = s;
			pix.j = t;
			(*pixels).push (pix); // a direita
			visited[s*width + t] = 1;
			(*tam_reg)++;
	    }
	}

	s = i + 1;
	t = j;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
			pix.i = s;
			pix.j = t;
			(*pixels).push (pix); // abaixo
			visited[s*width + t] = 1;
			(*tam_reg)++;
	    }
	}

	s = i;
	t = j - 1;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
			pix.i = s;
			pix.j = t;
			(*pixels).push (pix); // a esquerda
			visited[s*width + t] = 1;
			(*tam_reg)++;
	    }
	}
}

/****************************************************************************
 CCV Descriptor

 	Creates two histograms:
 		- histogram of coherent pixels
 		- histogram og incoherent pixels

 	Input
 		Mat original image
 		Mat histogram with size equal to 2*colors
 		int number of colors
 		int indicating if the normalization is requered
 		int level of coherency
 ****************************************************************************/
void CCV(Mat img, Mat *features, int colors, int normalization, int threshold){

	int i, j;
	Mat img_quant(img.size(), CV_8UC1);
	int height = img_quant.rows;
	int width = img_quant.cols;
	int *pxl_visited = new int[height*width]();
	long int tam_reg;
	vector<int> descriptor;

	for(i = 0; i < colors*2; i++){
		descriptor.push_back(0);
	}

	queue<Pixel> pixels;
	Pixel pix;

	if (img.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1)) / (max - min ));
		img_quant = img - min;
		img_quant = img_quant * stretch;
	}
	else {
		QuantizationMSB(img, &img_quant, colors);
	}

	// for each pixel, visit neighbors in order to find the regions
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			// if the current pixel was NOT already visited
			if (!pxl_visited[i*width + j]) {
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
					find_neighbor(img_quant, &pixels, pxl_visited, &tam_reg);

				// check if size of the region is higher than the coherence threshold
				if(tam_reg >= threshold)
					descriptor[pix.color*2 + 0] += tam_reg; // coerente
				  else
					descriptor[pix.color*2 + 1] += tam_reg; // incoerente
			}
		}
	}

	(*features).create(1, colors*2, CV_32F);
	(*features) = Scalar::all(0);

	if (normalization == 0) {
		for (i = 0; i < colors*2; i++) {
			(*features).at<float>(0,i) = (float)descriptor[i];
		}
	}
	else {
		float *norm = new float[colors*2];
		if (normalization == 1)
			NormalizeHist(&descriptor, norm, 2*colors, 1);
		else if (normalization == 2)
			NormalizeHist(&descriptor, norm, 2*colors, 255);
		for (i = 0; i < colors*2; i++) {
			(*features).at<float>(0,i) = norm[i];
		}
		delete[] norm;
	}

	delete[] pxl_visited;
	descriptor.clear();
}


/* Descritor GCH
 * Cria histrograma de cor da imagem.
 * Requer:
 *	- imagem original
 *	- histrograma ja alocado
 *	- quantidade de cores usadas na imagem */
void GCH(Mat I, Mat *features, int colors, int normalization) {

	int i;
	vector<int> hist;
	MatIterator_<uchar> it, end;

	Mat Q(I.size(), CV_8U, 1);
	if (I.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1)) / (max - min ));
		Q = I - min;
		Q = Q * stretch;
	}
	else {
		QuantizationMSB(I, &Q, colors);
	}

	for (i = 0; i < colors; i++) {
		hist.push_back(0);
	}

	end = Q.end<uchar>();
	for (it = Q.begin<uchar>(); it != end; it++) {
		hist[(*it)]++;
	}

	(*features).create(1, colors, CV_32F);
	(*features) = Scalar::all(0);

	if (normalization == 0) {
		for (i = 0; i < colors; i++) {
			(*features).at<float>(0,i) = (float)hist[i];
		}
	}
	else {
	    float *norm = new float[colors];
	    if (normalization == 1)
			NormalizeHist(&hist, norm, colors, 1);
	    else if (normalization == 2)
			NormalizeHist(&hist, norm, colors, 255);

	    for (i = 0; i < colors; i++) {
			(*features).at<float>(0,i) = norm[i];
	    }
	    delete[] norm;
	}
	hist.clear();
}


/* Descritor BIC
 * Cria dois histrogramas de cor da imagem:
 * 1 -> histograma de borda
 * 2 -> histograma de interior
 * Requer:
 *	- imagem original
 *	- histograma ja alocado, com tamanho de duas vezes a quantidade de cor
 *	- quantidade de cores usadas na imagem
 * No histograma, de 0 até (colors -1) = Borda, de colors até (2*colors -1) = Interior */
void BIC(Mat I, Mat *features, int colors, int normalization) {

	Size imgSize = I.size();
	int height = imgSize.height;
	int width = imgSize.width;
	int i, j;
	vector<int> hist;
	Mat Q(imgSize, CV_8U, 1);

	if (I.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1.0)) / (max - min ));
		Q = I - min;
		Q = Q * stretch;
	}
	else {
		QuantizationMSB(I, &Q, colors);
	}

	for (i = 0; i < 2*colors; i++) {
		hist.push_back(0);
	}

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			uchar aux = Q.at<uchar>(i,j);

			if (i > 0 && j > 0 && j < width-1 && i < height-1) {
				if ((Q.at<uchar>(i,j-1) == aux) &&
					(Q.at<uchar>(i,j+1) == aux) &&
					(Q.at<uchar>(i-1,j) == aux) &&
					(Q.at<uchar>(i+1,j) == aux)) {
					hist[aux]++;
				}
				else
					hist[aux+colors]++;
			}
			else
				hist[aux+colors]++;
		}
	}

	(*features).create(1, colors*2, CV_32F);
	(*features) = Scalar::all(0);

	if (normalization == 0) {
		for (i = 0; i < colors*2; i++) {
			(*features).at<float>(0,i) = (float)hist[i];
		}
	}
	else {
	    float *norm = new float[colors*2];
	    if (normalization == 1)
			NormalizeHist(&hist, norm, colors*2, 1);
	    else if (normalization == 2)
			NormalizeHist(&hist, norm, colors*2, 1024);

	    for (i = 0; i < colors*2; i++) {
			(*features).at<float>(0,i) = norm[i];
	    }
	    delete[] norm;
	}
	hist.clear();
}


/* Funcao CoocurrenceMatrix
 * Cria uma matriz que contem a ocorrencia de cada cor em cada pixel da imagem
 * Requer:
 *	- imagem original
 *	- matriz ja alocada, no tamanho colors x colors
 *	- quantidade de cores usadas na imagem
 *	- coordenadas dX e dY, que podem ser 0 ou 1 */
void CoocurrenceMatrix(Mat I, double **Cm, int colors, int dX, int dY) {

	int i, j, height, width;
	double sum = 0;

	Mat Q(I.size(), CV_8U, 1);
	if (I.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1)) / (max - min ));
		Q = I - min;
		Q = Q * stretch;
	}
	else {
		QuantizationMSB(I, &Q, colors);
	}

	Size newSize(I.rows+((dX+1)*2), I.cols+((dY+1)*2));
	Mat novaQ(newSize, CV_8U, 1);

	// copia a imagem quantizada e replica os pixels das bordas
	copyMakeBorder(Q, novaQ, (dX+1)*2, (dX+1)*2, (dY+1)*2, (dY+1)*2, BORDER_REPLICATE);

	Size imgSize = novaQ.size();
	height = imgSize.height;
	width = imgSize.width;

	// percorre a imagem
	for (i = dX; i < height-dX; i++) {
		for (j = dY; j < width-dY; j++) {
			int pixelRef = novaQ.at<uchar>(i,j);
			int pixelNeigh = novaQ.at<uchar>((i+dX),(j+dY));
			Cm[pixelRef][pixelNeigh]++;
		}
	}

	// calcula a soma das ocorrencias
	for (i = 0; i < colors; i++) {
		for (j = 0; j < colors; j++) {
			sum += Cm[i][j];
		}
	}

	// normaliza a matriz de forma que sum(Cm) = 1
	for (i = 0; i < colors; i++) {
		for (j = 0; j < colors; j++) {
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
void Haralick6(double **Cm, int colors, Mat *features) {

	int i,j;
	double m_r = 0.0, m_c = 0.0, s_r = 0.0, s_c = 0.0, entr = 0.0, auxv = 0.0;
	double maxp = 0.0, corr = 0.0, cont = 0.0, unif = 0.0, homo = 0.0;
	double *Pi = (double *) calloc(colors, sizeof(double));
	double *Pj = (double *) calloc(colors, sizeof(double));

	for (i = 0; i < colors; i++) {
		for (j = 0; j<colors; j++) {
			Pi[i] += Cm[i][j];
		}
		m_r += i*Pi[i];
	}

	for (j = 0; j < colors; j++) {
		for (i = 0; i<colors; i++) {
			Pj[j] += Cm[i][j];
		}
		m_c += j*Pj[j];
	}

	for (i = 0; i < colors; i++) {
		s_r += ((i-m_r)*(i-m_r)) * Pi[i];
		s_c += ((i-m_c)*(i-m_c)) * Pj[i];
	}
	s_r = sqrt(s_r);
	s_c = sqrt(s_c);

	for (i = 0; i < colors; i++) {
		for (j = 0; j<colors; j++) {
			auxv = Cm[i][j];

			// Descritor 1 - Maxima Probabilidade
			if (maxp < auxv) {
				maxp = auxv;
			}

			// Descritor 2 - correlacao
			if (s_r > 0 && s_c > 0) {
				corr += ((i-m_r)*(j-m_c)*auxv) / (s_r*s_c);
			}

			// Descritor 3 - contraste
			cont += ( (i-j)*(i-j)*auxv );

			// Descritor 4 - energia (uniformidade)
			unif += (auxv*auxv);

			// Descritor 5 - homogeneidade
			homo += (auxv) / (1 + abs(i-j));

			// Descritor 6 - entropia
			if (auxv != 0) {
				//entr+= auxv*( log(auxv) / log(2) );
				entr+= auxv*( log2(auxv));
			}
		}
	}

	(*features).create(1, 6, CV_32F);
	(*features) = Scalar::all(0);

	entr = -entr;
	(*features).at<float>(0, 0) = maxp;
	(*features).at<float>(0, 1) = corr;
	(*features).at<float>(0, 2) = cont;
	(*features).at<float>(0, 3) = unif;
	(*features).at<float>(0, 4) = homo;
	(*features).at<float>(0, 5) = entr;
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
void HARALICK(Mat I, double **Cm, Mat *features, int colors, int normalization) {
	CoocurrenceMatrix(I, Cm, colors, 2, 0);
	Haralick6(Cm, colors, features);
}


/* Descritor Autocorrelograma
 * Cria um histograma de cor que descreve a distribuição
 * global da correlação entre a localização espacial de cores
 * Requer:
 *	- imagem original
 *	- valor da distancia k entre os pixels
 *	- histograma ja alocado
 *	- quantidade de cores usadas na imagem */
void ACC(Mat I, Mat *features, int colors, int normalization, int *k, int totalk) {

	int i,j, x, y, maxdist, d, cd;
	vector<long int> desc(colors*totalk);
	double descNorm = 0;

	// aloca uma nova imagem do tamanho da original
	// com 8 bits por pixel e 1 canal de cor
	Mat Q(I.size(), CV_8U, 1);
	if (I.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1.0)) / (max - min));
		Q = I - min;
		Q = Q * stretch;
	}
	else {
		QuantizationMSB(I, &Q, colors);
	}

	Size imgSize = Q.size();
	int height = imgSize.height;// altura
	int width = imgSize.width;// largura

	// finds the maximum distance inside 'k'
	maxdist = 0;
	for (int d = 0; d < totalk; d++) {
	    if (k[d] > maxdist)
	    	maxdist = k[d];
	}

	// for each distance
	for (d = 0; d < totalk; d++) {
	    cd = k[d]; // current distance

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

	vector<float> norm(colors*totalk);
	float descsum = 0;
	for (i = 0; i < (colors*totalk) ; i++) {
		norm[i] = (float)(desc[i]/(float)descNorm);
		descsum += norm[i];
	}

	(*features).create(1, colors*totalk, CV_32F);
	(*features) = Scalar::all(0);

	if (normalization == 0) {
		for (i = 0; i < colors*totalk ; i++) {
			(*features).at<float>(0,i) = (float)desc[i];
		}
	}
	else if (normalization == 1) {
		for (i = 0; i < colors*totalk ; i++) {
			(*features).at<float>(0,i) = norm[i];
	  	}
	}
	else {
	    for (i = 0; i < colors*totalk; i++) {
			(*features).at<float>(0,i) = norm[i]*255;
	    }
	}
}

// Create a lookup table to check if it is uniform
vector<int> initUniform(){

    int index = 0, i = 0, b = 0, count = 0, c = 0;
    vector<int> lookup (255);

    for(i = 0; i < 256; i++) {

		b = (i >> 1) | (i << 7 & 0xff);
		c = i ^ b;
        //  Count the number of 1s in the binary representation
        for(count = 0; c; count++){
            c &= c-1; //clears the LSB
        }
        // Each uniform code is assigned to an index
        if (count <= 2) {
            lookup[i] = index;
            index++;
        }
        // All non uniform codes are assigned to 59
        else
            lookup[i]=57;
    }
    return lookup;
}

/****************************************************************************
 LBP Descriptor

 	It is a histogram of quantized LBPs pooled in a local image neighborhood.
	This version is an extension of the original LBP by using the proposed..

 	Input
 		Mat original image
 		Mat features vector in which perform the operations
 		int number of colors
 ****************************************************************************/
void LBP(Mat img, Mat *features, int colors){

	int bin, cellWidth, cellHeight, stride = 0, i, j, x, y, k, height, width;
	int increaseX = 1, increaseY = 1, newWidth, newHeight;
	int bitString;
	vector<int> lookup = initUniform();
	Size grid, cell;
	float center;
	height = img.rows;
	width = img.cols;
    Mat dst = Mat::zeros(img.rows, img.cols, CV_32FC1);

	Size newSize(width+increaseY*2, height+increaseX*2);
	Mat resizedImg(newSize, CV_8U, 1);

	Mat quantized(img.size(), CV_8U, 1);
	if (img.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1.0)) / (max - min ));
		quantized = img - min;
		quantized = quantized * stretch;
	}
	else {
		QuantizationMSB(img, &quantized, colors);
	}

	copyMakeBorder(quantized, resizedImg, increaseX, increaseX, increaseY, increaseY, BORDER_REPLICATE);

	Size imgSize = resizedImg.size();
	newHeight = imgSize.height;
	newWidth = imgSize.width;

	for (i = increaseY; i < newHeight - increaseY; i++) { // TODO: menor e igual?
		for (j = increaseX; j < newWidth - increaseX; j++) {

			/* For each pixel in a cell, compare the pixel to each of its 8 neighbors
				where the center pixel's value is greater than the neighbor's value, write "1". Otherwise, write "0".
			*/
			bitString = 0;
			center = img.at<uchar>(i,j);
			if(img.at<uchar>(i+1,j+0) >= center) bitString |= 0x1 << 0;
			if(img.at<uchar>(i+1,j+1) >= center) bitString |= 0x1 << 1;
			if(img.at<uchar>(i+0,j+1) >= center) bitString |= 0x1 << 2;
			if(img.at<uchar>(i-1,j+1) >= center) bitString |= 0x1 << 3;
			if(img.at<uchar>(i-1,j+0) >= center) bitString |= 0x1 << 4;
			if(img.at<uchar>(i-1,j-1) >= center) bitString |= 0x1 << 5;
			if(img.at<uchar>(i+0,j-1) >= center) bitString |= 0x1 << 6;
			if(img.at<uchar>(i+1,j-1) >= center) bitString |= 0x1 << 7;
			// This gives a bin corresponding to the binary code
			bin = lookup[bitString];
			dst.at<float>(i-increaseX, j-increaseY) = bin;
			// cout << " center " << center << " bitString " << bitString << " bin " << bin << endl;
        }
	}

	// Displays the lbp image
	// namedWindow("LBP Image", WINDOW_AUTOSIZE);
	// imshow("LBP Image", (dst/255.0)*4);
	// waitKey(0);

	Mat lbp = Mat::zeros(imgSize, CV_8U);
	copyMakeBorder(dst, lbp, 1, 1, 1, 1, BORDER_REPLICATE);

	grid.width = 2;
	grid.height = 2;
    cellWidth = newWidth/grid.width;
    cellHeight = newHeight/grid.height;

    int bias = 0;
    if (cellWidth*grid.width < width -1){
    	bias = 1;
    }
    stride = 0;

	(*features).create(1, 58*grid.width*grid.height, CV_32F);
	(*features) = Scalar::all(0);

    for(i = 0; i < grid.height; i++) {
        for(j = 0; j < grid.width; j++) {
            Mat cell = lbp(Rect(i*cellWidth+bias, j*cellHeight+bias, cellWidth, cellHeight));

            Mat cellHist = Mat::zeros(1, 58, CV_32FC1);

			for(x = 0; x < cellHeight; x++) {
				for(y = 0; y < cellWidth; y++) {
					bin = cell.at<float>(x,y);
					cellHist.at<float>(0,bin) += 1;
				}
			}

            for(k = 0; k < cellHist.cols; k++) {
                (*features).at<float>(0,(stride*58)+k) = cellHist.at<float>(0,k);
            }
            stride++;
        }
    }
}

/****************************************************************************
 Orientation Descriptor - Histogram of Oriented Gradients

 	Input
 		Mat original image
 		Mat features vector in which perform the operations
 ****************************************************************************/
void HOG(Mat img, Mat *features, int numFeatures){

	HOGDescriptor hog;
	vector<float> hogFeatures;
	vector<Point> locs;
	int i, cellSize, feat;

	Mat quantized(img.size(), CV_8U, 1);
	int colors = 256;
	if (img.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1.0)) / (max - min ));
		quantized = img - min;
		quantized = quantized * stretch;
	}
	else {
		QuantizationMSB(img, &quantized, colors);
	}

	/*numFeatures = hog.nbins * (blockSize.width/cellSize.width)
		* (blockSize.height/cellSize.height)
		* ((winSize.width - blockSize.width)/blockStride.width + 1)
		* ((winSize.height - blockSize.height)/ blockStride.height + 1); */
	int divideW = quantized.size().width / 8;
	int divideH = quantized.size().height / 8;
	hog.winSize = Size(divideW*8, divideH*8);
	cellSize = 8;
	if (numFeatures != 0)
		hog.nbins = numFeatures/((hog.winSize.width/cellSize-1) * (hog.winSize.height/cellSize-1) *2*2);
	hog.blockSize = Size(cellSize*2, cellSize*2);
	hog.blockStride = Size(cellSize, cellSize);
	hog.cellSize = Size(cellSize, cellSize);

	hog.compute(quantized,hogFeatures);

	feat = (numFeatures == 0) ? hogFeatures.size() : numFeatures;
	(*features).create(1, feat, CV_32F);
	(*features) = Scalar::all(0);

	for(i = 0; i < feat; i++){
		if ((int)hogFeatures.size() > i){
			(*features).at<float>(0,i) = hogFeatures.at(i);
		}
	}
}

/****************************************************************************
 Shape Descriptors - Contour Extraction

 	Input
 		Mat original image
 		Mat features vector in which perform the operations
 ****************************************************************************/
void contourExtraction(Mat img, Mat *features){

	vector<vector<Point> > contours;
	vector<Point> approx;
	vector<Vec4i> hierarchy;
	Mat bin;
	Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);
	int i, biggestAreaIndex = 0;
	bool k;
	double area, areaApprox, perimeter, biggestArea = 0;
	Moments mu;
	Point2f mc;

	Mat quantized(img.size(), CV_8U, 1);
	int colors = 256;
	if (img.channels() == 1) {
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((colors-1.0)) / (max - min ));
		quantized = img - min;
		quantized = quantized * stretch;
	}
	else {
		QuantizationMSB(img, &quantized, colors);
	}

	threshold(quantized, bin, 100, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	findContours(bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Mat contourImage(bin.size(), CV_8UC3, Scalar(0,0,0));
	for (i = 0; i < (int)contours.size(); i++) {
		// Scalar color(rand()&255, rand()&255, rand()&255);
		// drawContours(contourImage, contours, i, color);
		area = contourArea(contours[i], false);
		if (area > biggestArea){
			biggestArea = area;
			biggestAreaIndex = i;
		}
	}
	// namedWindow("All", 1);
	// imshow("All", contourImage);
	// waitKey(0);

	Scalar color(255, 255, 255);
	drawContours(contourImage, contours, biggestAreaIndex, color, 1, 8, hierarchy);
	// namedWindow("Biggest Contour", 1);
	// imshow("Biggest Contour", contourImage);
	// waitKey(0);

	(*features).create(1, 6, CV_32F);
	(*features) = Scalar::all(0);

	// Get the moments
	mu = moments(contours[biggestAreaIndex], false);
	//  Get the mass centers:
	mc = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
	(*features).at<float>(0,0) = mc.x;
	(*features).at<float>(0,1) = mc.y;

	// Number of pixels inside the contour
	(*features).at<float>(0,2) = biggestArea;

	// Contour perimeter
	perimeter = arcLength(contours[biggestAreaIndex], true);
	(*features).at<float>(0,3) = perimeter;

	// Remove small curves by approximating the contour more to a straight line
	approxPolyDP(contours[biggestAreaIndex], approx, 0.1*perimeter, true);
	areaApprox = contourArea(approx);
	(*features).at<float>(0,4) = areaApprox;

	// // Convex hull checks for convexity defects and corrects it
	// vector<int> hull;
	// convexHull(contours[biggestAreaIndex], hull); // returnPoints = true
	// vector<Point> hullPoints;
	// convexHull(contours[biggestAreaIndex], hullPoints);
	// k = isContourConvex(contours[biggestAreaIndex]);
	// (*features).at<float>(0,5) = k;
}

// /****************************************************************************
//  SURF - extract Speeded Up Robust Features
//
//  	Input
//  		Mat original image
//  		Mat features vector in which perform the operations
//  ****************************************************************************/
// void surf(Mat img, Mat *features){
//
// 	cv::initModule_nonfree();
// 	vector<KeyPoint> keypoints;
// 	int i, minHessian = 500;
//
// 	SurfFeatureDetector detector(minHessian);
// 	SurfDescriptorExtractor extractor;
//
// 	detector.detect(img, keypoints);
//
// 	Mat imgKey;
// 	drawKeypoints(img, keypoints, imgKey, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
// 	// imshow("Keypoints", imgKey);
// 	// waitKey(0);
// 	Mat descriptors;
// 	extractor.compute(img, keypoints, descriptors);
//
// 	cout << descriptors.size();
// 	(*features).create(descriptors.rows, 1, CV_32F);
// 	(*features) = Scalar::all(0);
//
// 	for(i = 0; i < descriptors.rows; i++){
// 		(*features).at<float>(0,i) = descriptors.at<float>(i,0);
// 	}
// }

/****************************************************************************
 Fisher Vectors Encoding

 	Input
 		Mat original image
 		Mat features vector in which perform the operations

	- Extract features with SIFT
	- Calculate a Gaussian Mixture Model (GMM)
	- Obtain the Fisher Vector encoding of a set of features
 ****************************************************************************/
void fisherVectors(Mat img, Mat *feature){
	float *means, *covariances, *priors, *posteriors, *enc;
	// int numClusters = 30;//, numData = img.cols,
	// int dimension = 3;

 	vector<KeyPoint> keypoints;
	Mat descriptors;
	float* data;
	vl_size numData, dimension, numClusters;
	// // create a GMM object and cluster input data to get means, covariances
	// // and priors of the estimated mixture
	// gmm = vl_gmm_new (VL_TYPE_FLOAT) ;
	// VLGMM** gmm;
	// vl_gmm_cluster (gmm, data, dimension, numData, numClusters);

	// // allocate space for the encoding
	// enc = vl_malloc(sizeof(float) * 2 * dimension * numClusters);
	// // run fisher encoding
	// vl_fisher_encode(
	// 	enc, VL_F_TYPE,
	// 	vl_gmm_get_means(gmm), dimension, numClusters,
	// 	vl_gmm_get_covariances(gmm),
	// 	vl_gmm_get_priors(gmm),
	// 	dataToEncode, numDataToEncode,
	// 	VL_FISHER_FLAG_IMPROVED
	// );
	//
	// 	(*features).create(enc.size(), 1, CV_32F);
	// 	(*features) = Scalar::all(0);
	//
	// 	for(i = 0; i < enc.size(); i++){
	// 		(*features).at<float>(0,i) = enc.at(i);
	// 	}
 }

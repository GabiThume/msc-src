/**
*
*	Author: Gabriela Thumé
*	Universidade de São Paulo / ICMC / 2014
*
**/

#include "preprocessing/artificialGeneration.h"

int classesNumber(string diretorio){

	struct dirent *sDir = NULL;
	DIR *dir = NULL;
	int count = 0;

	dir = opendir(diretorio.c_str());
	if(dir == NULL) {
		return 0;
	}

	while((sDir = readdir(dir))) {
		if( (strcmp(sDir->d_name, ".") != 0) &&
		(strcmp(sDir->d_name, "..") != 0) &&
		(strcmp(sDir->d_name, ".DS_Store") != 0) &&
		(strcmp(sDir->d_name, ".directory") != 0)) {
			count++;
		}
	}

	closedir(dir);
	return count;
}

Mat unsharp(Mat img, int N) {

	Mat out(img.size(), img.depth()), blur;
	int diff, i, j;

	GaussianBlur(img, blur, Size(N, N), 0);

	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			diff = img.at<uchar>(i,j) - blur.at<uchar>(i,j);
			out.at<uchar>(i,j) = saturate_cast<uchar>(img.at<uchar>(i,j) + diff);
		}
	}
	return out;
}

/* poissNoise(int lambda)
gera um valor ruidoso com base no valor passado por parametro
utilizando a distribuicao de Poisson.
o ruido eh correlacionado com o sinal e portanto
o nivel do ruido depende do nivel do sinal
*/
uchar poissNoise(int lambda){

	int k, value;
	double L, p;
	L = exp(-(double)lambda);
	p = 1.0;
	k = 0;
	do {
		k++;
		p = p * (rand()/(double)RAND_MAX);
	} while (p > L);

	value = saturate_cast<uchar>(k-1);
	return value;
}

Mat noiseSingleChannel(Mat img){
	int i, j;
	Mat out(img.size(), CV_8U);

	for(i = 0; i < img.rows; i++) {
		for(j = 0; j < img.cols; j++) {
			out.at<uchar>(i,j) = poissNoise((int)img.at<uchar>(i,j));
		}
	}
	return out;
}

Mat Artificial::generateNoise(Mat img) {

	Size s = img.size();
	Mat out(s, CV_8U, 3);

	vector<Mat> imColors(3);
	img.copyTo(out);

	split(out, imColors);

	imColors[0] = noiseSingleChannel(imColors[0]);
	imColors[1] = noiseSingleChannel(imColors[1]);
	imColors[2] = noiseSingleChannel(imColors[2]);

	merge(imColors, out);
	return out;
}

Mat Artificial::generateBlur(Mat originalImage, int blurType) {
	int i;
	Mat generated;

	// i = 3 + 2*(rand() % 15);
	i = 10 + (rand() % 140);
	//blurType = 1 + (rand() % 2);
	switch (blurType) {
		case 1:
			GaussianBlur(originalImage, generated, Size(i, i), 0);
			break;
		case 2:
			// bilateralFilter(originalImage, generated, i, i*2, i/2);
			bilateralFilter(originalImage, generated, 5, i, i);
			break;
		default:
			break;
	}
	return generated;
}

Mat Artificial::generateBlending(Mat originalImage, vector<Mat> images, int total){
	double alpha, beta;
	int randomSecondImg;
	Mat generated, second;

	alpha = 10.0 + (rand() % 80);
	beta = (100.0 - alpha);

	randomSecondImg = rand() % total;
	images[randomSecondImg].copyTo(second);
	if (originalImage.size() != second.size()) {
		resize(second, second, originalImage.size());
	}

	addWeighted(originalImage, alpha/100.0, second, beta/100.0, 0.0, generated);

	return generated;
}

Mat Artificial::generateUnsharp(Mat originalImage){
	int unsharpLevel;

	unsharpLevel = 3 + 2*(rand() % 5);

	Size s = originalImage.size();
	Mat generated(s, CV_8U, 3);

	vector<Mat> imColors(3);
	originalImage.copyTo(generated);

	split(generated, imColors);

	imColors[0] = unsharp(imColors[0], unsharpLevel);
	imColors[1] = unsharp(imColors[1], unsharpLevel);
	imColors[2] = unsharp(imColors[2], unsharpLevel);

	merge(imColors, generated);
	return generated;
}

Mat Artificial::generateComposition(Mat originalImage, vector<Mat> images,
		int total, int fator, bool option){

	vector<int> vectorRand;
	Mat subImg, generated, img, roi;
	originalImage.copyTo(subImg);
	int roiWidth, roiHeight, subImage, newImage, operation;
	int startWidth, startHeight, randH, randW;

	startWidth = startHeight = 0;
	for (subImage = 1; subImage <= fator; subImage++){
		do {
			newImage = rand() % total;
		} while(count(vectorRand.begin(), vectorRand.end(), newImage) &&
						(int)vectorRand.size() < total);
		vectorRand.push_back(newImage);
		images[newImage].copyTo(img);

		operation = 1 + (rand() % 4);
		switch(operation){
			case 1:
				generated = generateBlur(img, 2);
				break;
			case 2:
				generated = generateBlending(img, images, total);
				break;
			case 3:
				generated = generateUnsharp(img);
				break;
			case 4:
				generated = generateSaliency(img, images, total);
				break;
			case 5:
				generated = generateSmoteImg(img, images, total, 1);
				default:
			break;
		}
		roiHeight = subImg.size().height/sqrt(fator);
		roiWidth = subImg.size().width/sqrt(fator);

		if (generated.size().width < roiWidth ||
				generated.size().height < roiHeight){
			subImage--;
			continue;
		}
		if (option){
			randW = rand() % (generated.size().width - roiWidth);
			randH = rand() % (generated.size().height - roiHeight);
			generated(Rect(randW, randH, roiWidth, roiHeight)).copyTo(roi);
		}
		else{
			if (generated.size() == subImg.size()) {
				generated(Rect(startWidth, startHeight, roiWidth, roiHeight)).copyTo(roi);
			}
			else {
				generated(Rect(0, 0, roiWidth, roiHeight)).copyTo(roi);
			}
		}

		Mat dst_roi = subImg(Rect(startWidth, startHeight, roiWidth, roiHeight));
		roi.copyTo(dst_roi);
		if ((startWidth + 2*roiWidth) <= subImg.size().width){
			startWidth = startWidth + roiWidth;
		}
		else{
			startWidth = 0;
			startHeight = startHeight + roiHeight;
		}
		generated.release();
		img.release();
		roi.release();
		dst_roi.release();
	}
	return subImg;
}

Mat Artificial::generateThreshold(Mat originalImage, vector<Mat> images,
		int total){

	Mat generated, bin, foreground, background, second;
	int randomSecondImg;

	bilateralFilter(originalImage, bin, 6, 18, 3);
	//Create binary image using Otsu's threshold
	cvtColor(bin, bin, CV_BGR2GRAY);

	threshold(bin, bin, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);

	originalImage.copyTo(foreground, bin);

	// Select another image with the same size
	randomSecondImg = rand() % total;
	images[randomSecondImg].copyTo(second);
	if (originalImage.size() != second.size()) {
		resize(second, second, originalImage.size());
	}

	// namedWindow("Display window", WINDOW_AUTOSIZE );
	// imshow("original", originalImage);
	// imshow("foreground", bin);
	// imshow("second", randomSecondImg);

	// Select the background
	bitwise_not(bin, bin);
	second.copyTo(background, bin);

	// Blend of background and foreground
	generated = background + foreground;

	// imshow("background", background);
	// imshow("generated", generated);
	// waitKey(0);

	return generated;
}

Mat Artificial::generateSaliency(Mat originalImage, vector<Mat> images, int total){

	GMRsaliency GMRsal;
	Mat saliency_map, original, generated, bin, foreground, background, second;
	int randomSecondImg;
	originalImage.copyTo(original);
	saliency_map = GMRsal.GetSal(originalImage);

	while(original.size() != saliency_map.size()){
		original.release();
		saliency_map.release();
		images[rand() % total].copyTo(original);
		saliency_map = GMRsal.GetSal(original);
	}
	original.copyTo(bin);

	// Select just the most salient region, given a threshold value
	bin = saliency_map * 255;
	bin.convertTo(bin, CV_8U);
	threshold(bin, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	morphologyEx(bin, bin, MORPH_OPEN,
		getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1,1)), Point(-1,-1));
	original.copyTo(foreground, bin);

	// namedWindow( "Display window", WINDOW_AUTOSIZE );
	// imshow("saliency", bin);

	/* Select another image with the same size */
	randomSecondImg = rand() % total;
	images[randomSecondImg].copyTo(second);
	if (originalImage.size() != second.size()) {
		resize(second, second, originalImage.size());
	}

	// Select the background
	bitwise_not(bin, bin);
	second.copyTo(background, bin);

	// Blend of background and foreground
	generated = background + foreground;

	// imshow("saliency", generated);
	// waitKey(0);
	return generated;
}

Mat smoteImg(Mat first, Mat second, bool option){

	int i, j, newpixel, sizeHeight, sizeWidth;
	double diff, gap;

	sizeHeight = (first.size().height < second.size().height) ? first.size().height : second.size().height;
	sizeWidth = (first.size().width < second.size().width) ? first.size().width : second.size().width;
	Mat out(Size(sizeWidth, sizeHeight), CV_8U, 3);

	for (i = 0; i < sizeHeight; i++) {
		for (j = 0; j < sizeWidth; j++) {
			/* Calculate de difference between the same pixel in different images */
			diff = (int)first.at<uchar>(i,j) - (int)second.at<uchar>(i,j);
			/* Multiply this difference with a number between 0 and 1 */
			gap = (double)rand()/(RAND_MAX);
			newpixel = (int)first.at<uchar>(i,j);
			if (option)
				newpixel += gap*diff;
			else
				newpixel += (((newpixel + gap*diff) < 0) || ((newpixel + gap*diff) > 255) ) ? -gap*diff : gap*diff;
			out.at<uchar>(i,j) = saturate_cast<uchar>(newpixel);
		}
	}

	return out;
}

Mat Artificial::generateSmoteImg(Mat originalImage, vector<Mat> images, int total, bool option){

	int randomSecondImg;
	vector<Mat> imColors(3), imColorsSecond(3);
	Mat generated, second;

	randomSecondImg = rand() % total;
	images[randomSecondImg].copyTo(second);
	if (originalImage.size() != second.size()) {
		resize(second, second, originalImage.size());
	}

	originalImage.copyTo(generated);
	split(generated, imColors);
	split(second, imColorsSecond);

	imColors[0] = smoteImg(imColors[0], imColorsSecond[0], option);
	imColors[1] = smoteImg(imColors[1], imColorsSecond[1], option);
	imColors[2] = smoteImg(imColors[2], imColorsSecond[2], option);

	merge(imColors, generated);

	// namedWindow( "Display window", WINDOW_AUTOSIZE);
	// imshow("original", originalImage);
	// imshow("second", second);
	// imshow("smoteimg", generated);
	// waitKey(0);

	return generated;
}

void Artificial::GenerateImage(vector<Mat> images, int generationType,
		string name, Mat original, int total) {
	Mat generated;

	cout << "Generate " << name << " with operation " << generationType << endl;

	switch (generationType) {
		case 0: /* Replication */
			imwrite(name, original);
			break;
		case 2:
			generated = generateBlur(original, 2);
			imwrite(name, generated);
			break;
		case 3:
			generated = generateBlending(original, images, total);
			imwrite(name, generated);
			break;
		case 4:
			generated = generateUnsharp(original);
			imwrite(name, generated);
			break;
		case 5:
			generated = generateComposition(original, images, total, 16, 1);
			imwrite(name, generated);
			break;
		case 6:
			generated = generateThreshold(original, images, total);
			imwrite(name, generated);
			break;
		case 7:
			generated = generateSaliency(original, images, total);
			imwrite(name, generated);
			break;
		case 8:
			generated = generateSmoteImg(original, images, total, 1);
			imwrite(name, generated);
			break;
		case 9:
		    generated = generateNoise(original);
		    imwrite(name, generated);
		    break;
		// Those below this line are not used in our tests anymore
		case 10:
		    generated = generateSmoteImg(original, images, total, 0);
		    imwrite(name, generated);
		    break;
		case 11:
		    generated = generateComposition(original, images, total, 4, 0);
		    imwrite(name, generated);
		    break;
		case 12:
		    generated = generateComposition(original, images, total, 16, 1);
		    imwrite(name, generated);
		    break;
		case 13:
		    generated = generateComposition(original, images, total, 4, 1);
		    imwrite(name, generated);
		    break;
		case 14:
		    generated = generateComposition(original, images, total, 4, 2);
		    imwrite(name, generated);
		    break;
		case 15:
		    generated = generateComposition(original, images, total, 16, 2);
		    imwrite(name, generated);
		    break;
		default:
			break;
	}
	generated.release();
}

string Artificial::generate(string base, string newDirectory, int whichOperation = 0){

	int i, qtdClasses = 0, generationType, rebalanceTotal = 0;
	int maiorClasse, rebalance, eachClass, qtdImg, maior;
	Mat img, noise;
	string imgName, classe, minorityClass, str, nameGeneratedImage;
	struct dirent *sDir = NULL;
	DIR *dir = NULL, *minDir = NULL;
	vector<int> totalImage, vectorRand;
	vector<Mat> images;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(2, 9);

	dir = opendir(base.c_str());
	if (!dir) {
		cout << "Error! Directory " << base << " doesn't exist." << endl;
		exit(1);
	}
	closedir(dir);

	str = "rm -f -r "+newDirectory+"/*;";
	str += "mkdir -p "+newDirectory+";";
	str += "mkdir -p "+newDirectory+"/original/;";
	str += "cp -R "+base+"/* "+newDirectory+"/original/;";
	system(str.c_str());
	str = "rm -f -r "+newDirectory+"/features/;";
	str += "mkdir -p "+newDirectory+"/features/;";
	system(str.c_str());

	newDirectory += "/original/";
	dir = opendir(newDirectory.c_str());;
	if (!dir) {
		cout << "Error! Directory " << newDirectory << " doesn't exist. " << endl;
		exit(1);
	}

	cout << "\n---------------------------------------------------------" << endl;
	cout << "Artifical generation of images to rebalance classes" << endl;
	cout << "-----------------------------------------------------------" << endl;

	qtdClasses = classesNumber(newDirectory);
	cout << "Number of classes: " << qtdClasses << endl;
	closedir(dir);
	/* Count how many files there are in classes and which is the majority */
	maior = -1;
	for(i = 0; i < qtdClasses; i++) {
		classe = newDirectory + "/" + to_string(i) + "/";
		qtdImg = classesNumber(classe+"/treino/");
		if (qtdImg == 0){
			qtdImg = classesNumber(classe);
		}
		totalImage.push_back(qtdImg);
		if (qtdImg > maior){
			maiorClasse = i;
			maior = qtdImg;
		}
	}

	for(eachClass = 0; eachClass < qtdClasses; ++eachClass) {
		/* Find out how many samples are needed to rebalance */
		rebalance = totalImage[maiorClasse] - totalImage[eachClass];
		if (rebalance > 0) {

			minorityClass = newDirectory + "/" + to_string(eachClass) + "/treino/";
			cout << "Class: " << minorityClass << " contain " << totalImage[eachClass] << " images" << endl;
			minDir = opendir(minorityClass.c_str());
			if (!minDir) {
				cout << "Error! Directory " << minorityClass.c_str() << " doesn't exist. " << endl;
				exit(1);
			}
			/* Add all minority images in vector<Mat>images */
			while ((sDir = readdir(minDir))) {
				imgName = sDir->d_name;
				img = imread(minorityClass + imgName, CV_LOAD_IMAGE_COLOR);
				if (!img.data) continue;
				images.push_back(img);
			}
			if (images.size() == 0) {
				cout << "The class " << minorityClass+imgName << " could't be read" << endl;
				exit(-1);
			}
			closedir(minDir);

			/* For each image needed to full rebalance*/
			for (i = 0; i < rebalance; i++){
				/* Choose a random image */
				int randomImg = rand() % totalImage[eachClass];
				Mat original;
				images[randomImg].copyTo(original);
				/* Choose an operation
				Case 1: All operations */
				generationType = (whichOperation == 1) ? dis(gen) : whichOperation;

				nameGeneratedImage = minorityClass + to_string(totalImage[eachClass]+i) + ".png";
				GenerateImage(images, generationType, nameGeneratedImage, original,
					totalImage[eachClass]);
			}
			rebalanceTotal += rebalance;
			cout << rebalance << " images were generated and this is now balanced." << endl;
			cout << "-------------------------------------------------------" << endl;
			images.clear();
		}
	}
	return newDirectory;
}

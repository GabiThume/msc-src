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

/*******************************************************************************
Artificial generation: Unsharp Masking
*******************************************************************************/
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
/*******************************************************************************
Artificial generation: Poisson-distributed numbers (pseudo-random number sampling) by Knuth
*******************************************************************************/
uchar poissNoise(double lambda) {
	int k;
	double L, p;

	L = exp(-lambda);
	p = 1.0;
	k = 0;
	do {
		k++;
		p = p * (rand()/(double)RAND_MAX);
	} while (p > L);

	return saturate_cast<uchar>(k-1);
}

Mat noiseSingleChannel(Mat img){
	int i, j;
	Mat out(img.size(), CV_8U);

	for(i = 0; i < img.rows; i++) {
		for(j = 0; j < img.cols; j++) {
			out.at<uchar>(i,j) = poissNoise((double)img.at<uchar>(i,j));
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

/*******************************************************************************
Artificial generation: Blur
*******************************************************************************/
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
/*******************************************************************************
Artificial generation: Blending
*******************************************************************************/
Mat Artificial::generateBlending(Mat first, Mat second){
	double alpha, beta;
	Mat generated;

	alpha = 10.0 + (rand() % 80);
	beta = (100.0 - alpha);

	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}

	addWeighted(first, alpha/100.0, second, beta/100.0, 0.0, generated);

	return generated;
}

/*******************************************************************************
Artificial generation: Composition
*******************************************************************************/
Mat Artificial::generateComposition(Mat originalImage, vector<Mat> images,
		int total, int fator, bool option){

	vector<int> vectorRand;
	Mat subImg, generated, img, roi, second;
	originalImage.copyTo(subImg);
	int roiWidth, roiHeight, subImage, randomImg, operation, randomSecondImg;
	int startWidth, startHeight, randH, randW;

	startWidth = startHeight = 0;
	for (subImage = 1; subImage <= fator; subImage++){
		do {
			randomImg = rand() % total;
		} while(count(vectorRand.begin(), vectorRand.end(), randomImg) &&
						(int)vectorRand.size() < total);
		vectorRand.push_back(randomImg);
		images[randomImg].copyTo(img);

		randomSecondImg = randomImg;
		while (randomSecondImg == randomImg && total > 1) {
			randomSecondImg = rand() % total;
		}
		images[randomSecondImg].copyTo(second);

		operation = 1 + (rand() % 3);
		switch(operation){
			case 1:
				generated = generateBlur(img, 2);
				break;
			case 2:
				generated = generateBlending(img, second);
				break;
			case 3:
				generated = generateUnsharp(img);
				break;
			case 4:
				generated = generateSmoteImg(img, second);
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

/*******************************************************************************
Artificial generation: Threshold
*******************************************************************************/
Mat Artificial::generateThreshold(Mat first, Mat second) {

	Mat generated, bin, foreground, background;

	//Create binary image using Otsu's threshold
	cvtColor(first, bin, CV_BGR2GRAY);

	threshold(bin, bin, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	morphologyEx(bin, bin, MORPH_OPEN,
		getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1,1)), Point(-1,-1));

	morphologyEx(bin, bin, MORPH_DILATE,
		getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1,1)), Point(-1,-1));

	first.copyTo(foreground, bin);

	// Select the background
	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}
	bitwise_not(bin, bin);
	second.copyTo(background, bin);

	// Blend of background and foreground
	generated = background + foreground;

	// namedWindow("Display window", WINDOW_AUTOSIZE);
	// imshow("first", first);
	// imshow("foreground", foreground);
	// imshow("background", background);
	// imshow("generated", generated);
	// waitKey(0);

	return generated;
}

/*******************************************************************************
Artificial generation: Saliency
*******************************************************************************/
Mat Artificial::generateSaliency(Mat first, Mat second) {

	GMRsaliency GMRsal;
	Mat saliency_map, generated, bin, foreground, background;

	saliency_map = GMRsal.GetSal(first);
	if (first.size() != saliency_map.size()) {
		return Mat();
	}
	first.copyTo(bin);

	// Select just the most salient region, given a threshold value
	bin = saliency_map * 255;
	bin.convertTo(bin, CV_8U);
	threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	morphologyEx(bin, bin, MORPH_OPEN,
		getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1,1)), Point(-1,-1));
	morphologyEx(bin, bin, MORPH_DILATE,
		getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1,1)), Point(-1,-1));

	first.copyTo(foreground, bin);

	// namedWindow( "Display window", WINDOW_AUTOSIZE );
	// imshow("saliency", bin);

	// Select the background
	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}
	bitwise_not(bin, bin);
	second.copyTo(background, bin);

	// Blend of background and foreground
	generated = background + foreground;

	// imshow("saliency", generated);
	// waitKey(0);
	return generated;
}

/*******************************************************************************
Artificial generation: visual SMOTE
*******************************************************************************/
// Mat smoteImg(Mat first, Mat second){
//
// 	int i, j;
// 	double diff, gap, newpixel;
//
// 	if (first.size() != second.size()) {
// 		resize(second, second, first.size());
// 	}
// 	Mat out(first.size(), CV_8U, 1);
//
// 	for (i = 0; i < first.rows; i++) {
// 		for (j = 0; j < first.cols; j++) {
// 			/* Calculate de difference between the same pixel in different images */
// 			diff = abs((double)second.at<uchar>(i,j) - (double)first.at<uchar>(i,j));
// 			/* Multiply this difference with a number between 0 and 1 */
// 			gap = rand()/static_cast<double>(RAND_MAX);
// 			newpixel = (double)first.at<uchar>(i,j) + gap*diff;
// 			out.at<uchar>(i,j) = saturate_cast<uchar>(newpixel);
// 		}
// 	}
//
// 	return out;
// }

Mat smoteImg(Mat first, Mat second) {

	int i, j;
	double diff, gap;

	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}
	Mat out(first.size(), CV_64FC1, 1);

	for (i = 0; i < first.rows; i++) {
		for (j = 0; j < first.cols; j++) {
			/* Calculate de difference between the same pixel in different images */
			diff = (double)first.at<uchar>(i,j) - (double)second.at<uchar>(i,j);
			/* Multiply this difference with a number between 0 and 1 */
			gap = rand()/static_cast<double>(RAND_MAX);
			out.at<double>(i,j) = (double)first.at<uchar>(i,j) + gap*diff;
		}
	}

	reduceImageColors(&out, 256);
	out.convertTo(out, CV_8U);
	return out;
}

Mat Artificial::generateSmoteImg(Mat first, Mat second){

	vector<Mat> imColors(3), imColorsSecond(3);
	Mat generated;

	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}

	first.copyTo(generated);
	split(generated, imColors);
	split(second, imColorsSecond);

	imColors[0] = smoteImg(imColors[0], imColorsSecond[0]);
	imColors[1] = smoteImg(imColors[1], imColorsSecond[1]);
	imColors[2] = smoteImg(imColors[2], imColorsSecond[2]);

	merge(imColors, generated);

	// namedWindow( "Display window", WINDOW_AUTOSIZE);
	// imshow("original", originalImage);
	// imshow("second", second);
	// imshow("smoteimg", generated);
	// waitKey(0);

	return generated;
}

/*******************************************************************************
*******************************************************************************/
void Artificial::GenerateImage(vector<Mat> images, string name, int total,
															int generationType) {
	Mat generated, first, second;
	int randomImg, randomSecondImg;

	while (generated.empty()) {

		/* Choose a random image */
		randomImg = rand() % total;
		images[randomImg].copyTo(first);

		/* And a second one */
		randomSecondImg = randomImg;
		if (total > 1) {
			while (randomSecondImg == randomImg) {
				randomSecondImg = rand() % total;
			}
		}
		images[randomSecondImg].copyTo(second);

		cout << "Generate " << name << " with operation " << generationType << endl;

		switch (generationType) {
			case 0: /* Replication */
			  first.copyTo(generated);
				break;
			case 2:
				generated = generateBlur(first, 2);
				break;
			case 3:
				generated = generateBlending(first, second);
				break;
			case 4:
				generated = generateUnsharp(first);
				break;
			case 5:
				generated = generateComposition(first, images, total, 16, 1);
				break;
			case 6:
				generated = generateThreshold(first, second);
				break;
			case 7:
				generated = generateSaliency(first, second);
				break;
			case 8:
				generated = generateSmoteImg(first, second);
				break;
			case 9:
		    generated = generateNoise(first);
		    break;
			case 10:
		    generated = generateComposition(first, images, total, 4, 1);
		    break;
			default:
				break;
		}
	}
	if (generated.data) {
		imwrite(name, generated);
		generated.release();
	}
}

string Artificial::generate(string base, string newDirectory,
														int whichOperation = 0){

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
	std::uniform_int_distribution<> dis(2, 10);

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

	for (eachClass = 0; eachClass < qtdClasses; ++eachClass) {
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
			for (i = 0; i < rebalance; i++) {

				/* Choose an operation
				Case 1: All operations */
				generationType = (whichOperation == 1) ? dis(gen) : whichOperation;

				nameGeneratedImage = minorityClass + to_string(totalImage[eachClass]+i) + ".png";
				GenerateImage(images, nameGeneratedImage, totalImage[eachClass], generationType);
			}
			rebalanceTotal += rebalance;
			cout << rebalance << " images were generated and this is now balanced." << endl;
			cout << "-------------------------------------------------------" << endl;
			images.clear();
		}
	}
	return newDirectory;
}

/**
*
*	Author: Gabriela Thumé
*	Universidade de São Paulo / ICMC / 2014
*
**/

#include "preprocessing/artificialGeneration.h"

int classesNumber(std::string diretorio){

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
    Reduce the number of colors in a single channel image

    Requires:
    - I: input image to be modified
    - nColors: final number of colors
*******************************************************************************/
void reduceImageColors(cv::Mat *img, int nColors) {

	double min = 0, max = 0, stretch;
	cv::Point maxLoc, minLoc;
  int i, img_channels = (*img).channels();
  std::vector<cv::Mat> channel(img_channels);
  split((*img), channel);

  nColors = (nColors > 256) ? 256 : nColors;

  for (i = 0; i < img_channels; i++) {
    minMaxLoc(channel[i], &min, &max, &minLoc, &maxLoc);
    stretch = ((double)((nColors -1)) / (max - min));
    channel[i] = channel[i] - min;
    channel[i] = channel[i] * stretch;
  }

  merge(channel, (*img));
}

/*******************************************************************************
Artificial generation: Unsharp Masking
*******************************************************************************/
cv::Mat unsharp(cv::Mat img, int N) {

	cv::Mat out(img.size(), img.depth()), blur;
	int diff, i, j;

	GaussianBlur(img, blur, cv::Size(N, N), 0);

	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			diff = img.at<uchar>(i,j) - blur.at<uchar>(i,j);
			out.at<uchar>(i,j) = cv::saturate_cast<uchar>(img.at<uchar>(i,j) + diff);
		}
	}
	return out;
}

cv::Mat Artificial::generateUnsharp(cv::Mat originalImage){
	int unsharpLevel;

	unsharpLevel = 3 + 2*(rand() % 5);
	cv::Size s = originalImage.size();
	cv::Mat generated(s, CV_8U, 3);

	std::vector<cv::Mat> imColors(3);
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

	return cv::saturate_cast<uchar>(k-1);
}

cv::Mat noiseSingleChannel(cv::Mat img){
	int i, j;
	cv::Mat out(img.size(), CV_8U);

	for(i = 0; i < img.rows; i++) {
		for(j = 0; j < img.cols; j++) {
			out.at<uchar>(i,j) = poissNoise((double)img.at<uchar>(i,j));
		}
	}
	return out;
}

cv::Mat Artificial::generateNoise(cv::Mat img) {

	cv::Size s = img.size();
	cv::Mat out(s, CV_8U, 3);

	std::vector<cv::Mat> imColors(3);
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
cv::Mat Artificial::generateBlur(cv::Mat originalImage, int blurType) {
	int i;
	cv::Mat generated;

	// i = 3 + 2*(rand() % 15);
	i = 10 + (rand() % 140);
	//blurType = 1 + (rand() % 2);
	switch (blurType) {
		case 1:
			GaussianBlur(originalImage, generated, cv::Size(i, i), 0);
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
cv::Mat Artificial::generateBlending(cv::Mat first, cv::Mat second){
	double alpha, beta;
	cv::Mat generated;

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
cv::Mat Artificial::generateComposition(cv::Mat originalImage, std::vector<cv::Mat> images,
		int total, int fator, bool option){

	std::vector<int> vectorRand;
	cv::Mat subImg, generated, img, roi, second;
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
			generated(cv::Rect(randW, randH, roiWidth, roiHeight)).copyTo(roi);
		}
		else{
			if (generated.size() == subImg.size()) {
				generated(cv::Rect(startWidth, startHeight, roiWidth, roiHeight)).copyTo(roi);
			}
			else {
				generated(cv::Rect(0, 0, roiWidth, roiHeight)).copyTo(roi);
			}
		}

		cv::Mat dst_roi = subImg(cv::Rect(startWidth, startHeight, roiWidth, roiHeight));
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
cv::Mat Artificial::generateThreshold(cv::Mat first, cv::Mat second) {

	cv::Mat generated, bin, foreground, background;

	//Create binary image using Otsu's threshold
	cvtColor(first, bin, CV_BGR2GRAY);

	threshold(bin, bin, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	morphologyEx(bin, bin, MORPH_OPEN,
		getStructuringElement(MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1,1)), cv::Point(-1,-1));

	morphologyEx(bin, bin, MORPH_DILATE,
		getStructuringElement(MORPH_CROSS, cv::Size(3, 3), cv::Point(1,1)), cv::Point(-1,-1));

	first.copyTo(foreground, bin);

	// Select the background
	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}
	bitwise_not(bin, bin);
	second.copyTo(background, bin);

	// Blend of background and foreground
	generated = background + foreground;

	// cv::namedWindow("Display window", WINDOW_AUTOSIZE);
	// imshow("first", first);
	// imshow("foreground", foreground);
	// imshow("background", background);
	// imshow("generated", generated);
	// cv::waitKey(0);

	return generated;
}

/*******************************************************************************
Artificial generation: Saliency
*******************************************************************************/
cv::Mat Artificial::generateSaliency(cv::Mat first, cv::Mat second) {

	GMRsaliency GMRsal;
	cv::Mat saliency_map, generated, bin, foreground, background;

	saliency_map = GMRsal.GetSal(first);
	if (first.size() != saliency_map.size()) {
		return first;
	}
	first.copyTo(bin);

	// Select just the most salient region, given a threshold value
	bin = saliency_map * 255;
	bin.convertTo(bin, CV_8U);
	threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	morphologyEx(bin, bin, MORPH_OPEN,
		getStructuringElement(MORPH_CROSS, cv::Size(3, 3), cv::Point(1,1)), cv::Point(-1,-1));
	morphologyEx(bin, bin, MORPH_DILATE,
		getStructuringElement(MORPH_CROSS, cv::Size(3, 3), cv::Point(1,1)), cv::Point(-1,-1));

	first.copyTo(foreground, bin);

	// cv::namedWindow( "Display window", WINDOW_AUTOSIZE );
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
	// cv::waitKey(0);
	return generated;
}

/*******************************************************************************
Artificial generation: visual SMOTE
*******************************************************************************/
// cv::Mat smoteImg(cv::Mat first, cv::Mat second){
//
// 	int i, j;
// 	double diff, gap, newpixel;
//
// 	if (first.size() != second.size()) {
// 		resize(second, second, first.size());
// 	}
// 	cv::Mat out(first.size(), CV_8U, 1);
//
// 	for (i = 0; i < first.rows; i++) {
// 		for (j = 0; j < first.cols; j++) {
// 			/* Calculate de difference between the same pixel in different images */
// 			diff = abs((double)second.at<uchar>(i,j) - (double)first.at<uchar>(i,j));
// 			/* Multiply this difference with a number between 0 and 1 */
// 			gap = rand()/static_cast<double>(RAND_MAX);
// 			newpixel = (double)first.at<uchar>(i,j) + gap*diff;
// 			out.at<uchar>(i,j) = cv::saturate_cast<uchar>(newpixel);
// 		}
// 	}
//
// 	return out;
// }

cv::Mat smoteImg(cv::Mat first, cv::Mat second) {

	int i, j;
	double diff, gap;

	if (first.size() != second.size()) {
		resize(second, second, first.size());
	}
	cv::Mat out(first.size(), CV_64FC1, 1);

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

cv::Mat Artificial::generateSmoteImg(cv::Mat first, cv::Mat second){

	std::vector<cv::Mat> imColors(3), imColorsSecond(3);
	cv::Mat generated;

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

	// cv::namedWindow( "Display window", WINDOW_AUTOSIZE);
	// imshow("original", originalImage);
	// imshow("second", second);
	// imshow("smoteimg", generated);
	// cv::waitKey(0);

	return generated;
}

/*******************************************************************************
*******************************************************************************/
void Artificial::GenerateImage(std::vector<cv::Mat> images, std::string name,
	int total, int generationType) {

	cv::Mat generated, first, second;
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

		std::cout << "Generate " << name;
		std::cout << " with operation " << generationType << std::endl;

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

void Artificial::generateImagesFromData(Data *originalData,
		std::string newDirectory, int whichOperation) {

	int i, generationType, rebalanceTotal = 0, biggest, trainingImagesInClass;
	int rebalance, j, training_fold, generatedFold;
	cv::Mat img, noise;
	std::string imgName, classe, minorityClass, str, nameGeneratedImage, generatedPath;
	DIR *dir = NULL, *minDir = NULL;
	std::vector<int> vectorRand;
	std::vector<cv::Mat> images;
	std::vector<ImageClass>::iterator itClass;
	ImageClass thisClass;
	Image newImage;

	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Artifical generation of images to rebalance classes";
	std::cout << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;

	str = "rm -f -r "+newDirectory+"/*;";
	str += "mkdir -p "+newDirectory+";";
	system(str.c_str());
	dir = opendir(newDirectory.c_str());;
	if (!dir) {
		std::cout << "Error! Directory " << newDirectory;
		std::cout << " can't be created. " << std::endl;
		exit(1);
	}
	closedir(dir);

	// str += "mkdir -p "+newDirectory+"/original/;";
	// str += "cp -R "+base+"/* "+newDirectory+"/original/;";
	str = "rm -f -r "+newDirectory+"/features/;";
	str += "mkdir -p "+newDirectory+"/features/;";
	system(str.c_str());
	dir = opendir((newDirectory+"/features/").c_str());;
	if (!dir) {
		std::cout << "Error! Directory " << newDirectory << " can't be created. " << std::endl;
		exit(1);
	}
	closedir(dir);

	std::cout << "Number of classes: " << (*originalData).numClasses() << std::endl;

	biggest = (*originalData).biggestTrainingNumber();

	for (itClass = (*originalData).classes.begin();
			itClass != (*originalData).classes.end();
			++itClass) {
		/* Find out how many samples are needed to rebalance */
		trainingImagesInClass = (*originalData).numTrainingImages(itClass->id);
		rebalance = biggest - trainingImagesInClass;
		if (rebalance > 0) {

			/* Add all minority images in std::vector<cv::Mat>images */
			for(j = 0; j < (int) itClass->images.size(); j++) {
				for (training_fold = 0;
						training_fold < (int) itClass->training_fold.size(); training_fold++) {
					if (itClass->images[j].fold == training_fold) {
						imgName = itClass->images[j].path;
						img = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
						if (!img.data) continue;
						images.push_back(img);
					}
				}
			}

			if (images.size() == 0) {
				std::cout << "The class " << itClass->id << " could't be read\n";
				exit(-1);
			}

			std::stringstream generatedClass;
		 	generatedClass << itClass->id;
			generatedPath = newDirectory + generatedClass.str() + "/";
			minDir = opendir(generatedPath.c_str());
			if (!minDir) {
				str = "mkdir -p "+generatedPath+";";
				std::cout << str << std::endl;
				system(str.c_str());
				minDir = opendir(generatedPath.c_str());
				if (!minDir) {
					std::cout << "Error! Directory " << generatedPath.c_str() << " can't be created." << std::endl;
					exit(1);
				}
			}
			closedir(minDir);

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(2, 10);

			generatedFold = (*originalData).newFold(itClass->id);

			/* For each image needed to full rebalance*/
			for (i = 0; i < rebalance; i++) {

				/* Choose an operation
				Case 1: All operations */
				generationType = (whichOperation == 1) ? dis(gen) : whichOperation;

				nameGeneratedImage = generatedPath + std::to_string(trainingImagesInClass+i) + ".png";

				GenerateImage(images, nameGeneratedImage, trainingImagesInClass, generationType);

				newImage.features.release();
				newImage.fold = generatedFold;
				newImage.generationType = generationType;
				newImage.path = nameGeneratedImage;
				itClass->images.push_back(newImage);
			}
			itClass->generated_fold.push_back(generatedFold);
			itClass->training_fold.push_back(generatedFold);

			rebalanceTotal += rebalance;
			std::cout << rebalance << " images were generated and the class ";
			std::cout << itClass->id << " is now balanced." << std::endl;
			std::cout << "---------------------------------------------" << std::endl;
			images.clear();
		}
	}
}

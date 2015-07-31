/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "artificialGeneration.h"

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

    Mat out(img.size(), img.depth());
    Mat blur;
    int diff, newpixel;

    GaussianBlur(img, blur, Size(N, N), 0);

    for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++) {
    
            diff = img.at<uchar>(i,j) - blur.at<uchar>(i,j);
            newpixel = img.at<uchar>(i,j) + diff;
            // overflow / underflow control
            newpixel = (newpixel > 255) ? 255 : newpixel;
            newpixel = (newpixel < 0) ? 0 : newpixel;

            out.at<uchar>(i,j) = newpixel;  
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
int poissNoise(int lambda){

    int k, value;
    double L, p;
    L = exp(-(double)lambda);
    p = 1.0;
    k = 0;
    do {
        k++;
        p = p * (rand()/(float)RAND_MAX);
    } while (p > L);

    value = k-1;
    value = (value > 255) ? 255 : value;
    return value;
}

Mat noiseSingleChannel(Mat img){

    int height, width, cinzar, cinza, i, j;
    Mat out(img.size(), CV_8U);
    
    height = img.size().height;
    width = img.size().width;

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            cinza = (int)img.at<uchar>(i,j);
            cinzar = poissNoise(cinza);
            out.at<uchar>(i,j) = (uchar)cinzar;
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

Mat Artificial::generateBlur(Mat originalImage){
    int i, blurType;
    Mat generated;

    i = 3 + 2*(rand() % 15);
    blurType = 1 + (rand() % 2);
    switch (blurType) {
        case 1: 
            GaussianBlur(originalImage, generated, Size(i, i), 0);
            break;
        case 2:
            bilateralFilter(originalImage, generated, i, i*2, i/2);
            break;
    }
    return generated;
}

Mat Artificial::generateBlending(Mat originalImage, vector<Mat> images, int total){
    float alpha, beta;
    int randomSecondImg;
    Mat generated;

    alpha = (rand() % 100);
    beta = (100.0 - alpha);
    randomSecondImg = 0 + (rand() % total);
    while (originalImage.size() != images[randomSecondImg].size()){
        randomSecondImg = 0 + (rand() % total);
    }
    addWeighted(originalImage, alpha/100.0, images[randomSecondImg], beta/100.0, 0.0, generated);
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

Mat Artificial::generateComposition(Mat originalImage, vector<Mat> images, int total, int fator){

    vector<int> vectorRand;
    Mat subImg, generated, img, roi;
    originalImage.copyTo(subImg);
    int roiWidth, roiHeight, subImage, newImage, operation;
    int startWidth, startHeight;
    
    startWidth = startHeight = 0;
    for (subImage = 1; subImage <= fator; subImage++){
        // vectorRand.clear();
        // do{
        //     newImage = rand() % total;
        // } while(count(vectorRand.begin(), vectorRand.end(), newImage));
        // vectorRand.push_back(newImage);

        newImage = rand() % total;
        /* Find out if the subimage has the same size */
        images[newImage].copyTo(img);
        // if (img.size() != subImg.size()){
        //     subImage--;
        //     continue;
        // }
        img.copyTo(generated);

        operation = 1 + (rand() % 5);
        switch(operation){
            case 1:
                generated = generateBlur(originalImage);
                break;
            case 2:
                generated = generateNoise(originalImage);
                break;
            case 3:
                generated = generateBlending(originalImage, images, total);
                break;
            case 4:
                generated = generateUnsharp(originalImage);
                break;
            case 5:
                generated = generateThreshold(originalImage, images, total);
                break;
            case 6:
                generated = generateSaliency(originalImage, images, total);
            default:
                break;
        }

        roiHeight = generated.size().height/sqrt(fator);
        roiWidth = generated.size().width/sqrt(fator);

        if (img.size() == subImg.size()){
            generated(Rect(startWidth, startHeight, roiWidth, roiHeight)).copyTo(roi);
        }
        else{
            generated(Rect(0, 0, roiWidth, roiHeight)).copyTo(roi);
        }
        Mat dst_roi = subImg(Rect(startWidth, startHeight, roiWidth, roiHeight));
        roi.copyTo(dst_roi);
        if ((startWidth + roiWidth) < generated.size().width){
            startWidth = startWidth + roiWidth;
        }
        else{
            startWidth = 0;
            startHeight = startHeight + roiHeight;
        }
    }
    return subImg;
}

Mat Artificial::generateThreshold(Mat originalImage, vector<Mat> images, int total){

    Mat generated, bin, foreground, background, saliency_map;
    int randomSecondImg;

    //Create binary image using Otsu's threshold
    cvtColor(originalImage, bin, CV_BGR2GRAY);
    threshold(bin, bin, 0, 255, CV_THRESH_OTSU);

    // // Select the foreground
    //erode(bin,bin,Mat(),Point(-1,-1), 2, 1, 1);
    //dilate(bin,bin,Mat(),Point(-1,-1), 2, 1, 1);
    // MORPH_RECT
    //morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(2*2+1, 2*2+1), Point(2,2)), Point(-1,-1)); 
    originalImage.copyTo(foreground, bin&1);

    // namedWindow("Display window", WINDOW_AUTOSIZE );
    // imshow("foreground", bin); 
    // waitKey(0);

    // Select another image with the same size
    randomSecondImg = 0 + (rand() % total);
    while (originalImage.size() != images[randomSecondImg].size()){
        randomSecondImg = 0 + (rand() % total);
    }

    // Select the background
    bitwise_not(bin, bin);
    images[randomSecondImg].copyTo(background, bin);

    // namedWindow("Display window", WINDOW_AUTOSIZE );
    // imshow("background", background); 
    // waitKey(0);

    // Blend of background and foreground 
    generated = background + foreground;

    // namedWindow("Display window", WINDOW_AUTOSIZE );
    // imshow("generated", generated); 
    // waitKey(0);

    // originalImage.copyTo(bin); 
    // // Select just the most salient region, given a threshold value
    // bin = saliency_map * 255;
    // GaussianBlur(bin, bin, Size(1,1), 0, 0);
    // bin.convertTo(bin, CV_8U); // threshold needs an int Mat

    return generated;
}

Mat Artificial::generateSaliency(Mat originalImage, vector<Mat> images, int total){

    GMRsaliency GMRsal;
    Mat saliency_map, original, generated, bin, foreground, background;
    int randomSecondImg;
    originalImage.copyTo(original);
    saliency_map = GMRsal.GetSal(originalImage);

    while(original.size() != saliency_map.size()){
        images[rand() % total].copyTo(original);
        saliency_map = GMRsal.GetSal(original);
    }
    original.copyTo(bin); 

    // Select just the most salient region, given a threshold value
    bin = saliency_map * 255;
    GaussianBlur(bin, bin, Size(1,1), 0, 0);
    bin.convertTo(bin, CV_8U); // threshold needs an int Mat
    threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Eliminate small regions (Mat() == default 3x3 kernel)
    // morphologyEx(bin, bin, 3, getStructuringElement(2, Size( 2*20 + 1, 2*20+1 ), Point(20, 20)));
    original.copyTo(foreground, bin&1); 

    // imwrite(nameGeneratedImage+"_saliency", bin);
    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow("saliency", bin); 
    // waitKey(0);

    /* Select another image with the same size */
    randomSecondImg = rand() % total;
    while (original.size() != images[randomSecondImg].size()){
        randomSecondImg = rand() % total;
    }

    // Select the background
    bitwise_not(bin, bin);
    images[randomSecondImg].copyTo(background, bin);

    // Blend of background and foreground 
    generated = background + foreground;

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow("saliency", generated); 
    // waitKey(0);
    return generated;
}

void Artificial::generate(string base, int whichOperation = 0){

	int i, qtdClasses = 0, generationType, rebalanceTotal = 0;
    int maior = -1, maiorClasse, rebalance, qtdImg, eachClass;
    Mat img, noise;
    string imgName, classe, minorityClass;
	struct dirent *sDir = NULL;
    DIR *dir = NULL, *minDir = NULL;
    vector<int> totalImage, vectorRand;
    vector<Mat> images;
    srand(time(0));

	dir = opendir(base.c_str());
	if(dir == NULL) {
		cout << "Error! Directory " << base << " don't exist. " << endl;
		exit(1);
	}

    cout << "\n---------------------------------------------------------------------------------------" << endl;
    cout << "Artifical generation of images to rebalance classes" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;

    qtdClasses = classesNumber(base);
    cout << "Number of classes: " << qtdClasses << endl;
    /* Count how many files there are in classes and which is the majority */
    for(i = 0; i < qtdClasses; i++) {
        classe = base + "/" + to_string(i) + "/";
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
        if (rebalance > 0){

            minorityClass = base + "/" + to_string(eachClass) + "/treino/";
            cout << "Class: " << minorityClass << " contain " << totalImage[eachClass] << " images" << endl;
            minDir = opendir(minorityClass.c_str());

            /* Add all minority images in vector<Mat>images */
        	while((sDir = readdir(minDir))) {
        	    imgName = sDir->d_name;
                img = imread(minorityClass + imgName, CV_LOAD_IMAGE_COLOR);
                if (!img.data) continue;
                images.push_back(img);
            }

            /* For each image needed to full rebalance*/
            for (i = 0; i < rebalance; i++){
                /* Choose a random image */
                int randomImg = 0 + (rand() % totalImage[eachClass]);
                Mat generated, subImg, tmp, bin, foreground, background;

                /* Choose an operation 
                    Case 1: All operation */
                if (whichOperation == 1)          
                    generationType = 2 + (rand() % 8);
                else
                    generationType = whichOperation;

                string nameGeneratedImage = minorityClass + to_string(totalImage[eachClass]+i) + ".jpg";

                switch (generationType) {
                    case 0: /* Replication */
                        imwrite(nameGeneratedImage, images[randomImg]);
                        break;
                    case 2:
                        generated = generateBlur(images[randomImg]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 3:
                        generated = generateNoise(images[randomImg]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 4:
                        generated = generateBlending(images[randomImg], images, totalImage[eachClass]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 5:
                        generated = generateUnsharp(images[randomImg]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 6:
                        generated = generateComposition(images[randomImg], images, totalImage[eachClass], 4);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 7:
                        generated = generateComposition(images[randomImg], images, totalImage[eachClass], 16);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 8:
                        generated = generateThreshold(images[randomImg], images, totalImage[eachClass]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    case 9:
                        generated = generateSaliency(images[randomImg], images, totalImage[eachClass]);
                        imwrite(nameGeneratedImage, generated);
                        break;
                    default:
                        break;
                }
                generated.release();
            }
            rebalanceTotal += rebalance;
            cout << rebalance << " images were generated and this is now balanced." << endl;
            cout << "---------------------------------------------------------------------------------------" << endl;
            images.clear();
        }
    }
}

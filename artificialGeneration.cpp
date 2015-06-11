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
		// cout << "Erro!! Nao foi possivel abrir o diretorio " << diretorio << endl;
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

Mat unsharpmask(Mat img, int N) {

    Size s = img.size();
    Mat out(s, CV_8U, 3);

    vector<Mat> imColors(3);
    img.copyTo(out);
    
    split(out, imColors);

    imColors[0] = unsharp(imColors[0], N);
    imColors[1] = unsharp(imColors[1], N);
    imColors[2] = unsharp(imColors[2], N);

    merge(imColors, out);
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

Mat generateNoise(Mat img) {

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


int Artificial::generate(string base, int isToGenerate = 1, int whichOperation = 0){

	int i, j, qtdClasses = 0, generationType, roiHeight, roiWidth;
    int operation, imageOrigin, subImage, randomSecondImg, unsharpLevel, blurType;
    int menor = 999, maior = -1, maiorClasse, menorClasse, rebalance, qtdImg;
    double alpha, beta;
    Mat img, noise;
    string imgName, classe, minorityClass;
	struct dirent *sDir = NULL;
    DIR *dir = NULL, *minDir = NULL;
    vector<int> totalImage, vectorRand;
    vector<Mat> images;
    srand(time(0));
    GMRsaliency GMRsal;
    Mat saliency_map;

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
    /* Count how many files there are in classes and which is
        minority or majority */
    for(i = 1; i <= qtdClasses; i++) {
        classe = base + "/" + to_string(i) + "/";
        qtdImg = classesNumber(classe+"/treino/");
        if (qtdImg == 0){
           qtdImg = classesNumber(classe);
        }
        totalImage.push_back(qtdImg);
        if (qtdImg < menor){
            menorClasse = i;
            menor = qtdImg;
        }
        if (qtdImg > maior){
            maiorClasse = i;
            maior = qtdImg;
        }
    }
    /* Find how many samples it is needed to rebalance */
    //TODO: dividido por 2? igual a classe de treino da majoritaria
    rebalance = totalImage[maiorClasse-1]/2 - totalImage[menorClasse-1];
    if (rebalance == 0){
        cout << "Error! Classes were already balanced.\n" << endl;
        exit(-1);
    }
    cout << "totalImage[maiorClasse-1]/2 " << totalImage[maiorClasse-1]/2 << " totalImage[menorClasse-1] " << totalImage[menorClasse-1] << endl;

    minorityClass = base + "/" + to_string(menorClasse) + "/treino/";
    cout << "Minority Class: " << minorityClass << endl;
    minDir = opendir(minorityClass.c_str());

    /* Add all minority images in vector<Mat>images */
	while((sDir = readdir(minDir))) {
	    imgName = sDir->d_name;
        img = imread(minorityClass + imgName, CV_LOAD_IMAGE_COLOR);
        if (!img.data){
            continue;
        }
        images.push_back(img);
    }

    /* For each image needed to full rebalance*/
    for (i = 0; i < rebalance; i++){
        /* Choose a random image */
        int randomImg = 0 + (rand() % totalImage[menorClasse-1]);
        Mat generated, subImg, tmp, bin, foreground, background;
        images[randomImg].copyTo(generated);
        images[randomImg].copyTo(subImg);

        /* Choose an operation 
            Case 1: All operation */
        if (whichOperation == 1)          
            generationType = 2 + (rand() % 7);
        else
            generationType = whichOperation;

        string nameGeneratedImage = minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg";

        // height = images[randomImg].size().height;
        // width = images[randomImg].size().width;
        // cout << " type " << generationType << " name " << nameGeneratedImage << endl;
        switch (generationType) {

        case 0: /* Replication */
            imwrite(nameGeneratedImage, images[randomImg]);
            break;
        case 2: /* Blurring */
            j = 3 + 2*(rand() % 15);
            blurType = 1 + (rand() % 2);
            switch (blurType) {
                case 1: 
                    GaussianBlur(images[randomImg], generated, Size(j, j), 0);
                    break;
                case 2:
                    bilateralFilter(images[randomImg], generated, j, j*2, j/2);
                    break;
            }
            imwrite(nameGeneratedImage, generated);
            break;
        case 3: /* Apply noise */
            generated = generateNoise(images[randomImg]);
            imwrite(nameGeneratedImage, generated);
            break;
        case 4: /* blending */
            alpha = (rand() % 100);
            beta = (100.0 - alpha);
            randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            while (images[randomImg].size() != images[randomSecondImg].size()){
                randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            }
            addWeighted(images[randomImg], alpha/100.0, images[randomSecondImg], beta/100.0, 0.0, generated);
            imwrite(nameGeneratedImage, generated);
            break;
        case 5: 
            unsharpLevel = 3 + 2*(rand() % 5);
            generated = unsharpmask(images[randomImg], unsharpLevel);
            imwrite(nameGeneratedImage, generated);
            break;
        case 6:
            vectorRand.clear();
            images[randomImg].copyTo(subImg);

            for (subImage = 1; subImage <= 4; subImage++){
                vectorRand.clear();
                do{
                    imageOrigin = (rand() % (totalImage[menorClasse-1]+i));
                } while(count(vectorRand.begin(), vectorRand.end(), imageOrigin));
                vectorRand.push_back(imageOrigin);

                /* Find out if the subimage has the same size */
                img = imread(minorityClass + to_string(imageOrigin) + ".jpg", CV_LOAD_IMAGE_COLOR);
                if (img.size() != subImg.size()){
                    subImage--;
                    continue;
                }
                img.copyTo(generated);

                /* Apply blur or noise in the subimage */
                operation = 1 + (rand() % 4);
                switch(operation){
                    case 1: /* Blurring */
                        j = 3 + 2*(rand() % 15);
                        blurType = 1 + (rand() % 2);
                        switch (blurType) {
                            case 1: 
                                GaussianBlur(images[randomImg], generated, Size(j, j), 0);
                                break;
                            case 2:
                                bilateralFilter(images[randomImg], generated, j, j*2, j/2);
                                break;
                        }
                        break;
                    case 2: /* Apply noise */
                        generated = generateNoise(img);
                        break;
                    case 3: /* blending */
                        alpha = (rand() % 100);
                        beta = (100.0 - alpha);
                        randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                        while (img.size() != images[randomSecondImg].size()){
                            randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                        }
                        addWeighted(img, alpha/100.0, images[randomSecondImg], beta/100.0, 0.0, generated);
                        break;
                    case 4: /* unsharp masking */
                        unsharpLevel = 3 + 2*(rand() % 5);
                        generated = unsharpmask(img, unsharpLevel);
                        break;
                    default:
                        break;
                }

                roiHeight = generated.size().height/2;
                roiWidth = generated.size().width/2;
                if (subImage == 1){
                    Mat roi = generated(Rect(0, 0, roiWidth, roiHeight));
                    Mat dst_roi = subImg(Rect(0, 0, roiWidth, roiHeight));
                    roi.copyTo(dst_roi);
                }
                else if (subImage == 2){
                    Mat roi = generated(Rect(roiWidth, 0, roiWidth, roiHeight));
                    Mat dst_roi = subImg(Rect(roiWidth, 0, roiWidth, roiHeight));
                    roi.copyTo(dst_roi);
                }
                else if (subImage == 3){
                    Mat roi = generated(Rect(0, roiHeight, roiWidth, roiHeight));
                    Mat dst_roi = subImg(Rect(0, roiHeight, roiWidth, roiHeight));
                    roi.copyTo(dst_roi);
                }
                else if (subImage == 4){
                    Mat roi = generated(Rect(roiWidth, roiHeight, roiWidth, roiHeight));
                    Mat dst_roi = subImg(Rect(roiWidth, roiHeight, roiWidth, roiHeight));
                    roi.copyTo(dst_roi);
                }
            }
            imwrite(nameGeneratedImage, subImg);
            break;
        case 7: /* Threshold */

            // Create binary image using Otsu's threshold
            // cvtColor(images[randomImg], bin, CV_BGR2GRAY);
            // threshold(bin, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

            // // Select the foreground
            // erode(bin,bin,Mat(),Point(-1,-1), 2, 1, 1);
            // dilate(bin,bin,Mat(),Point(-1,-1), 2, 1, 1);
            // // morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(2*2+1, 2*2+1), Point(2,2)), Point(-1,-1), i ); 
            // images[randomImg].copyTo(foreground, bin&1);

            // // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
            // // imshow( "foreground", bin); 
            // // waitKey(0);

            // // Select another image with the same size
            // randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            // while (images[randomImg].size() != images[randomSecondImg].size()){
            //     randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            // }

            // // Select the background
            // bitwise_not(bin, bin);
            // images[randomSecondImg].copyTo(background, bin);

            // // Blend of background and foreground 
            // generated = background + foreground;

            // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
            // imshow( "foreground", generated); 
            // waitKey(0);

             images[randomImg].copyTo(bin); 
                // Select just the most salient region, given a threshold value
                bin = saliency_map * 255;
                GaussianBlur(bin, bin, Size(1,1), 0, 0);
                bin.convertTo(bin, CV_8U); // threshold needs an int Mat
                threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

                // Eliminate small regions (Mat() == default 3x3 kernel)
                // morphologyEx(bin, bin, 3, getStructuringElement(2, Size( 2*20 + 1, 2*20+1 ), Point(20, 20)));
                images[randomImg].copyTo(foreground, bin&1); 

                // imwrite(nameGeneratedImage+"_saliency", bin);
                // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
                // imshow("saliency", bin); 
                // waitKey(0);

                // Select another image with the same size
                randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                while (images[randomImg].size() != images[randomSecondImg].size()){
                    randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                }

                // Select the background
                bitwise_not(bin, bin);
                images[randomSecondImg].copyTo(background, bin);

                // Blend of background and foreground 
                generated = background + foreground;
                
            imwrite(nameGeneratedImage, generated);
            break;
        case 8: /* Salient region */
            saliency_map = GMRsal.GetSal(images[randomImg]);

            if (images[randomImg].size() != saliency_map.size()){
                i--;
                continue;
            }
            else{

                images[randomImg].copyTo(bin); 
                // Select just the most salient region, given a threshold value
                bin = saliency_map * 255;
                GaussianBlur(bin, bin, Size(1,1), 0, 0);
                bin.convertTo(bin, CV_8U); // threshold needs an int Mat
                threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

                // Eliminate small regions (Mat() == default 3x3 kernel)
                // morphologyEx(bin, bin, 3, getStructuringElement(2, Size( 2*20 + 1, 2*20+1 ), Point(20, 20)));
                images[randomImg].copyTo(foreground, bin&1); 

                // imwrite(nameGeneratedImage+"_saliency", bin);
                // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
                // imshow("saliency", bin); 
                // waitKey(0);

                // Select another image with the same size
                randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                while (images[randomImg].size() != images[randomSecondImg].size()){
                    randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                }

                // Select the background
                bitwise_not(bin, bin);
                images[randomSecondImg].copyTo(background, bin);

                // Blend of background and foreground 
                generated = background + foreground;

                // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
                // imshow("saliency", generated); 
                // waitKey(0);

                imwrite(nameGeneratedImage, generated);
            }
            break;
        default:
            break;
        }
        generated.release();
        subImg.release();
        tmp.release();
        bin.release(); 
        foreground.release(); 
        background.release();
    }
    cout << rebalance << " images were generated and the minority class is now balanced." << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    return rebalance;
}

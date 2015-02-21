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

int Artificial::generate(string base, int isToGenerate = 1, int whichOperation = 0){

	int i, j, qtdClasses = 0, generationType, noiseType, roiHeight, roiWidth;
    int operation, imageOrigin, subImage, randomSecondImg, noiseLevel, unsharpLevel;
    int menor = 999, maior = -1, maiorClasse, menorClasse, rebalance, qtdImg;
    float weight;
    double alpha, beta;
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
    rebalance = totalImage[maiorClasse-1]/2 - totalImage[menorClasse-1];
    if (rebalance == 0){
        cout << "Error! Classes were already balanced.\n" << endl;
        exit(-1);
    }

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

        Mat generated, subImg, tmp;
        images[randomImg].copyTo(generated);
        images[randomImg].copyTo(subImg);

        /* Choose an operation */
        if (whichOperation == 0)
            generationType = 1 + (rand() % 4);
        else if(whichOperation == -1){
            imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", images[randomImg]);
            continue;
        }
        else
            generationType = whichOperation;
        //isToGenerate = 0;
        //generationType = 5;
        string nameGeneratedImage = minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg";

        switch (generationType) {

        case 1: /* Blurring */
            // j = 3 + 2*(rand() % 5);
            // // j = 5;
            // alpha = (rand() % 100);
            // GaussianBlur(images[randomImg], generated, Size(j, j), 0, 0);
            // bilateralFilter(images[randomImg], generated, j, j*2, j/2);
            j = 3 + 2*(rand() % 15);
            GaussianBlur(images[randomImg], generated, Size(j, j), 0);
            // bilateralFilter(images[randomImg], generated, 15, 80, 80);
            imwrite(nameGeneratedImage, generated);
            break;
        case 2: /* Apply noise */
            noise = generated.clone();
            noiseLevel = 5 + (rand() % 20);
            // randn(noise, 0, 20);
            randn(noise, 0, noiseLevel);
            // j = 3 + 2*(rand() % 15);
            GaussianBlur(noise, noise, Size(3, 3), 0);
            add(generated, noise, generated);
            imwrite(nameGeneratedImage, generated);
            break;
        case 3: /* blending */
            alpha = (rand() % 100);
            beta = (100.0 - alpha);
            randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            while (images[randomImg].size().height != images[randomSecondImg].size().height){
                randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
            }
            addWeighted(images[randomImg], alpha/100.0, images[randomSecondImg], beta/100.0, 0.0, generated);
            imwrite(nameGeneratedImage, generated);
            break;
        // case 4: /* unsharp masking */
        //     // unsharpLevel = 3 + 2*(rand() % 5);
        //     // unsharpLevel = 1;
        //     // weight = (1 + (rand() % 3))/2;
        //     GaussianBlur(images[randomImg], generated, Size(5, 5), 0);
        //     addWeighted(images[randomImg], 1.5, generated, -0.5, 0, generated);
        //     imwrite(nameGeneratedImage, generated);
        //     break;
        case 4:
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
                if (img.size().height != subImg.size().height){
                    subImage--;
                    continue;
                }
                img.copyTo(generated);

                /* Apply blur or noise in the subimage */
                operation = 1 + (rand() % 3);
                switch(operation){
                    case 1: /* Blurring */
                        j = 3 + 2*(rand() % 15);
                        alpha = (rand() % 100);
                        GaussianBlur(img, generated, Size(j, j), 0);
                        // bilateralFilter(img, generated, j, j*2, j/2);
                        break;
                    case 2: /* Apply noise */
                        noise = generated.clone();
                        noiseLevel = 1 + 2*(rand() % 5);
                        randn(noise, 0, noiseLevel);
                        GaussianBlur(noise, noise, Size(3, 3), 0.5);
                        add(generated, noise, generated);
                        break;
                    case 3: /* blending */
                        alpha = (rand() % 100);
                        beta = (100.0 - alpha);
                        randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                        while (img.size().height != images[randomSecondImg].size().height){
                            randomSecondImg = 0 + (rand() % totalImage[menorClasse-1]);
                        }
                        addWeighted(img, alpha/100.0, images[randomSecondImg], beta/100.0, 0.0, generated);
                        break;
                    // case 4: /* unsharp masking */
                    //     unsharpLevel = 3 + 2*(rand() % 5);
                    //     GaussianBlur(images[randomImg], generated, Size(5, 5), 5);
                    //     addWeighted(img, 1.5, generated, -0.5, 0, generated);
                    //     break;
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
        default:
            break;
        }
    }
    cout << rebalance << " images were generated and the minority class is now balanced." << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    return rebalance;
}

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
		cout << "Erro!! Nao foi possivel abrir o diretorio " << diretorio << endl;
		return -1;
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

void Artificial::generate(string base, int isToGenerate = 1){

	int i, j, qtdClasses = 0, generationType, noiseType, roiHeight, roiWidth;
    int operation, imageOrigin, subImage;
    int menor = 999, maior = -1, maiorClasse, menorClasse, rebalance, qtdImg;
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
        qtdImg = classesNumber(classe);
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
    rebalance = totalImage[maiorClasse-1] - totalImage[menorClasse-1];
    if (rebalance == 0){
        cout << "Error! Classes were already balanced.\n" << endl;
        exit(-1);
    }

    minorityClass = base + "/" + to_string(menorClasse) + "/";
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

        Mat generated, subImg;
        images[randomImg].copyTo(generated);
        images[randomImg].copyTo(subImg);

        /* Choose an operation */
        generationType = 1 + (rand() % 3);
        switch (generationType) {

        case 1: /* Blurring */
            if(isToGenerate == 0){
                imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", images[randomImg]);
                break;
            }
            j = 3 + 2*(rand() % 15);
            GaussianBlur(images[randomImg], generated, Size(j, j), 0, 0);
            imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", generated);
            break;
        case 2: /* Apply noise */
            if(isToGenerate == 0){
                imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", images[randomImg]);
                break;
            }
            noise = generated.clone();
            randn(noise, 0, 20);
            GaussianBlur(noise, noise, Size(3, 3), 0.5, 0.5);
            add(generated, noise, generated);
            /*for (x = 0; x < height; x++) {
                for (y = 0; y < width; y++) {
                    cinza = (int)images[i].at<uchar>(x,y);
                    cinzar = poissNoise(cinza);
                    generated.at<uchar>(x,y) = (uchar)cinzar;
                }
            }*/
            /*namedWindow( "noise", WINDOW_AUTOSIZE );
            imshow( "noise", generated );
            waitKey(0);*/
            imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", generated);
            break;
        case 3:
            if(isToGenerate == 0){
                imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", images[randomImg]);
                break;
            }
            vectorRand.clear();
            images[randomImg].copyTo(subImg);

            for (subImage = 1; subImage <= 4 ; subImage++){
                vectorRand.clear();
                do{
                    imageOrigin = noiseType = (rand() % (totalImage[menorClasse-1]+i));
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
                operation = 1 + (rand() % 2);
                switch(operation){
                    case 1:
                        j = 3 + 2*(rand() % 15);
                        GaussianBlur(img, generated, Size(j, j), 0, 0);
                        break;
                    case 2:
                        noise = generated.clone();
                        randn(noise, 0, 20);
                        GaussianBlur(noise, noise, Size(3, 3), 0.5, 0.5);
                        add(generated, noise, generated);
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
            imwrite(minorityClass + to_string(totalImage[menorClasse-1]+i) + ".jpg", subImg);
            break;
//        case 4: // blending
//            break;
//        case 5: // unsharp mask
//            break;
        default:
            break;
        }
    }
    cout << rebalance << " images were generated and the minority class is now balanced." << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
}

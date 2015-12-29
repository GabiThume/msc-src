/*******************************************************************************
Generate a imbalanced class
*******************************************************************************/
std::string Rebalance::RemoveSamples(std::string database, std::string newDir, double prob, double id) {

  int pos = 0, samples, imagesTraining, i, imgsInClass, x, qtdClasses;
  int prev, count_diff, treino, num_images, imagesTesting;
  std::vector<int> vectorRand, objperClass;
  std::string str, nameFile, name, nameDir, directory, dir;
  std::stringstream numImages, classNumber, image, globalFactor;
  cv::Size size;
  std::ifstream myFile;
  cv::Mat data, classes;
  double fator = 1.0;

  // Check how many classes and images there are
  directory = database+"/";
  qtdClasses = qtdArquivos(directory);
  if (qtdClasses < 2) {
    std::cout << "Error. There is less than two classes in " << directory << std::endl;
    exit(-1);
  }
  NumberImgInClass(directory, 0, &prev, &treino);
  imagesTesting = prev;
  count_diff = 0;
  for (i = 1; i < qtdClasses; i++) {
    NumberImgInClass(database, i, &num_images, &treino);
    if (num_images < imagesTesting) imagesTesting = num_images;
    count_diff = count_diff + abs(num_images - prev);
    prev = num_images;
  }
  imagesTesting = imagesTesting*prob;
  if (count_diff == 0) {
    std::cout << "\n\n------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Divide the number of original samples to create a minority class:" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
  }

  globalFactor << id;
  dir = newDir+"/Imbalance-"+globalFactor.str()+"/";
  str = "rm -f -r "+dir+"*;"+"mkdir -p "+dir+";";
  str += "cp -r "+database+"* "+dir+";";
  system(str.c_str());

  for(i = 0; i < qtdClasses; i++) {
    classNumber.str("");
    classNumber << i;
    directory = database + "/" + classNumber.str()  + "/";
    imgsInClass = qtdArquivos(directory);
    if (imgsInClass == 0){
      fprintf(stderr,"Error! There is no directory named %s\n", directory.c_str());
      exit(-1);
    }
    samples = ceil((1.0 + fator)*imgsInClass*prob);
    if (samples > 0){

      numImages.str("");
      numImages << vectorRand.size();

      //str = "rm "+dir+classNumber.str()+"/*;";
      str = "rm -f -r "+dir+classNumber.str()+"/*;";
      str += "mkdir -p "+dir+classNumber.str()+"/treino/;";
      str += "mkdir -p "+dir+classNumber.str()+"/teste/;";

      //std::cout << " Executa " << str.c_str() << std::endl;
      system(str.c_str());

      /* Generate a random position to select samples to create the minority class */
      while ((int)vectorRand.size() < samples) {
        pos = rand() % imgsInClass;
        if (!count(vectorRand.begin(), vectorRand.end(), pos)){
          vectorRand.push_back(pos);
        }
      }

      /* Copy some of the originals to a training folder */
      // imagesTraining = vectorRand.size()*prob;
      imagesTraining = vectorRand.size() - imagesTesting;
      for(x = 0; x < imagesTraining; x++){
        image.str("");
        image << vectorRand[x];
        str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
        str+= dir+classNumber.str()+"/treino/; ";
        system(str.c_str());
        //std::cout << " Executa " << str.c_str() << std::endl;
      }

      /* Copy the rest of originals to a testing folder */
      for(x = imagesTraining; x < (int)vectorRand.size(); x++){
        image.str("");
        image << vectorRand[x];
        str = "cp "+database+classNumber.str()+"/"+image.str()+".png ";
        str+= dir+classNumber.str()+"/teste/;";
        system(str.c_str());
        //std::cout << " Executa " << str.c_str() << std::endl;
      }

      str = "bash scripts/rename.sh "+dir+classNumber.str()+"/";
      system(str.c_str());

      vectorRand.clear();
    }
    if (count_diff == 0) {
      // fator -= id/(double)qtdClasses;
      if (id != 1.0) {
        fator -= (1.0 - id);
      } else {
        fator -= 1.0/(double)qtdClasses;
      }
    }
  }
  return dir;
}

std::string Rebalance::PerformFeatureExtraction(std::string database, std::string featuresDir, int method,
    int colors, double resizeFactor, int normalization, std::vector<int> param,
    int deleteNull, int quantization, std::string id){

  int numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0;
  int i, j, bars, current_class;
  double porc;
  int resizingFactor = static_cast<int>(resizeFactor*100);
  std::string name, directory;
  cv::Mat img, featureVector, features, labels, trainTest, newimg, isGenerated;
  std::vector<int> num_images_class;
  std::vector<string> path;
  clock_t begin, end;

  std::cout << "\n---------------------------------------------------------" << std::endl;
  std::cout << "Image feature extraction using " << descriptorMethod[method-1];
  std::cout << " and " << quantizationMethod[quantization-1] << std::endl;
  std::cout << "-----------------------------------------------------------" << std::endl;

  std::cout << "Database: " << database << std::endl;

  begin = clock();
  img = cv::imread(database, CV_LOAD_IMAGE_COLOR);
  if (!img.empty()) {
    // Resize the image given the input factor
    resize(img, newimg, cv::Size(), resizeFactor, resizeFactor, INTER_AREA);

    // Convert the image to grayscale
    ConvertToGrayscale(quantization, newimg, &newimg, colors);

    // Call the description method
    GetFeatureVector(method, newimg, &features, colors, normalization, param);
    if (featureVector.cols == 0) {
      std::cout << "Error: the feature std::vector is null" << std::endl;
      exit(1);
    }

    img.release();
    newimg.release();
  } else {
    // Check how many classes and images there are
    qtdClasses = qtdArquivos(database+"/");
    qtdImgTotal = NumberImagesInDataset(database, qtdClasses, &num_images_class);
    labels = cv::Mat::zeros(qtdImgTotal, 1, CV_32S);
    trainTest = cv::Mat::zeros(qtdImgTotal, 1, CV_32S);
    isGenerated= cv::Mat::zeros(qtdImgTotal, 1, CV_32S);


    for (i = 0; i < qtdClasses; i++) {
      NumberImgInClass(database, i, &numImages, &treino);

      for (j = 0; j < numImages; j++)    {
        // The image label is the i index
        labels.at<int>(imgTotal, 0) = i;
        // Find this image in the class and open it
        img = FindImgInClass(database, i, j, imgTotal, treino, &trainTest, &path,
                            &isGenerated);
        if (!img.empty()) {

          // Resize the image given the input size
          img.copyTo(newimg);
          if (resizeFactor != 1.0) {
            cv::resize(img, newimg, cv::Size(), resizeFactor, resizeFactor, INTER_AREA);
          }

          // Convert the image to grayscale
          ConvertToGrayscale(quantization, newimg, &newimg, colors);

          // Call the description method
          GetFeatureVector(method, newimg, &featureVector, colors, normalization,
            param);
          if (featureVector.cols == 0) {
            std::cout << "Error: the feature std::vector is null" << std::endl;
            exit(1);
          }

          // Push the feature std::vector for the current image in the features std::vector
          features.push_back(featureVector);
          imgTotal++;
          featureVector.release();
          img.release();
          newimg.release();
        }
      }
    }
    // Normalization of Haralick and contourExtraction features by z-index
    if ((method == 4 || method == 8) && normalization) {
      ZScoreNormalization(&features);
    }

    // Remove null columns in cv::Mat of features
    if (deleteNull) {
      RemoveNullColumns(&features);
    }
  }

  // Show the number of images per class
  std::cout << "Images: " << features.rows << " - Classes: " << qtdClasses;
  std::cout << " - Features: " << features.cols << std::endl;
  for (current_class = 0; current_class < qtdClasses; current_class++) {
    bars = (static_cast<double> (num_images_class[current_class]) /
      static_cast<double> (features.rows)) * 50.0;
    std::cout << current_class << " ";
    for (j = 0; j < bars; j++) {
      std::cout << "|";
    }
    porc = static_cast<double> (num_images_class[current_class]) /
      static_cast<double> (features.rows);
    std::cout << " " << porc * 100.0 << "%" << " (";
    std::cout << num_images_class[current_class] << ")" <<std::endl;
  }

  name = WriteFeaturesOnFile(featuresDir, quantization, method, colors,
    normalization, resizingFactor, qtdClasses, features, labels, trainTest,
    isGenerated, path, id, false);
  std::cout << "File: " << name << std::endl;
  end = clock();
  std::cout << std::endl << "Elapsed time: " << double(end-begin)/ CLOCKS_PER_SEC << std::endl;

  return name;
}

int NumberImagesInDataset(std::string base, int qtdClasses,
  std::vector<int> *objClass) {

  int i, count = 0, currentSize;
  std::string directory;

  for (i = 0; i < qtdClasses; i++) {
    directory = base + "/" + std::to_string(i) + "/treino/";
    currentSize = qtdArquivos(directory);
    directory = base + "/" + std::to_string(i) + "/treino/generated/";
    currentSize += qtdArquivos(directory);
    directory = base + "/" + std::to_string(i) + "/teste/";
    currentSize += qtdArquivos(directory);
    if (currentSize == 0) {
      directory = base + "/" + std::to_string(i)  + "/";
      currentSize = qtdArquivos(directory);
      if (currentSize == 0) {
        std::cout << "Error: there is no directory named " << directory.c_str();
        exit(-1);
      }
    }
    (*objClass).push_back(currentSize);
    count += currentSize;
  }
  return count;
}

/*******************************************************************************
Counts how many images are in a class, and how many of those are for training

Requires
- std::string database name
- int class number
- int* output the total number of images
- int* output the number of images in training
*******************************************************************************/
void NumberImgInClass(std::string database, int img_class, int *num_imgs,
  int *num_train) {

  std::string directory;

  directory = database + "/" + std::to_string(img_class)  + "/treino/";
  (*num_imgs) = qtdArquivos(directory);
  directory = database + "/" + std::to_string(img_class)  + "/treino/generated/";
  (*num_imgs) += qtdArquivos(directory);
  (*num_train) = (*num_imgs);

  directory = database + "/" + std::to_string(img_class)  + "/teste/";
  (*num_imgs) += qtdArquivos(directory);

  if ((*num_imgs) == 0) {
    directory = database + "/" + std::to_string(img_class)  + "/";
    (*num_imgs) = qtdArquivos(directory);

    if ((*num_imgs) == 0) {
      std::cout << "Error: there is no directory named " << directory.c_str();
      exit(-1);
    }
  }
  std::cout << "class " << img_class << ": " << database + "/" + std::to_string(img_class);
  std::cout << " has " << (*num_imgs) << " images" << std::endl;
}

cv::Mat FindImgInClass(std::string database, int img_class, int img_number,
  int index, int treino, cv::Mat *trainTest, std::vector<std::string> *path, cv::Mat *isGenerated) {

  std::string directory, dir_class = database +"/" + std::to_string(img_class);
  cv::Mat img;

  directory = dir_class + "/"+std::to_string(img_number);
  img = cv::imread(directory+".png", CV_LOAD_IMAGE_COLOR);

  // std::cout << directory+".png" << std::endl;
  (*trainTest).at<int>(index, 0) = 0;
  (*isGenerated).at<int>(index, 0) = 0;

  if (img.empty()) {
    directory = dir_class + "/treino/" + std::to_string(img_number);
    img = cv::imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
    (*trainTest).at<int>(index, 0) = 1;

    if (img.empty()) {
      directory = dir_class + "/treino/generated/" + std::to_string(img_number);
      img = cv::imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        directory = dir_class + "/teste/" + std::to_string(img_number - treino);
        img = cv::imread(directory+".png", CV_LOAD_IMAGE_COLOR);
        (*trainTest).at<int>(index, 0) = 2;

        if (img.empty()) {
          std::cout << "Error: there is no image in " << directory.c_str();
          return cv::Mat();
        }
      } else {
        (*isGenerated).at<int>(index, 0) = 1;
      }
    }
  }
  (*path).push_back(directory + ".png");
  return img;
}

std::vector<std::vector<double> > Rebalance::classify(
  std::string descriptorFile, int repeat, double prob, std::string csv) {

  Classifier c;
  int minoritySize, numClasses;
  std::vector<std::vector<double> > fscore;
  // Read the feature vectors
  std::vector<ImageClass> data = ReadFeaturesFromFile(descriptorFile);
  numClasses = data.size();
  if (numClasses != 0){
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Features vectors file: " << descriptorFile.c_str() << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    c.findSmallerClass(data, &minoritySize);
    fscore = c.classify(prob, repeat, data, csv.c_str(), minoritySize);
    data.clear();
  }
  return fscore;
}

std::string Artificial::generate(std::string base, std::string newDirectory,
	int whichOperation = 0) {

	int i, qtdClasses = 0, generationType, rebalanceTotal = 0;
	int maiorClasse, rebalance, eachClass, qtdImg, maior;
	cv::Mat img, noise;
	std::string imgName, classe, minorityClass, str, nameGeneratedImage, generatedPath;
	struct dirent *sDir = NULL;
	DIR *dir = NULL, *minDir = NULL;
	std::vector<int> totalImage, vectorRand;
	std::vector<cv::Mat> images;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(2, 10);

	dir = opendir(base.c_str());
	if (!dir) {
		std::cout << "Error! Directory " << base << " doesn't exist." << std::endl;
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
		std::cout << "Error! Directory " << newDirectory << " doesn't exist. " << std::endl;
		exit(1);
	}

	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Artifical generation of images to rebalance classes";
	std::cout << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;

	qtdClasses = classesNumber(newDirectory);
	std::cout << "Number of classes: " << qtdClasses << std::endl;
	closedir(dir);
	/* Count how many files there are in classes and which is the majority */
	maior = -1;
	for(i = 0; i < qtdClasses; i++) {
		classe = newDirectory + "/" + std::to_string(i) + "/";
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

			minorityClass = newDirectory + "/" + std::to_string(eachClass) + "/treino/";
			std::cout << "Class: " << minorityClass << " contain " << totalImage[eachClass] << " images" << std::endl;
			minDir = opendir(minorityClass.c_str());
			if (!minDir) {
				std::cout << "Error! Directory " << minorityClass.c_str() << " doesn't exist. " << std::endl;
				exit(1);
			}
			/* Add all minority images in std::vector<cv::Mat>images */
			while ((sDir = readdir(minDir))) {
				imgName = sDir->d_name;
				img = cv::imread(minorityClass + imgName, CV_LOAD_IMAGE_COLOR);
				if (!img.data) continue;
				images.push_back(img);
			}
			if (images.size() == 0) {
				std::cout << "The class " << minorityClass+imgName << " could't be read" << std::endl;
				exit(-1);
			}
			closedir(minDir);

			generatedPath = minorityClass + "/generated/";
			minDir = opendir(generatedPath.c_str());
			if (!minDir) {
				str = "mkdir -p "+generatedPath+";";
				system(str.c_str());
				minDir = opendir(generatedPath.c_str());
				if (!minDir) {
					std::cout << "Error! Directory " << generatedPath.c_str() << " can't be created. " << std::endl;
					exit(1);
				}
			}
			closedir(minDir);

			/* For each image needed to full rebalance*/
			for (i = 0; i < rebalance; i++) {

				/* Choose an operation
				Case 1: All operations */
				generationType = (whichOperation == 1) ? dis(gen) : whichOperation;

				nameGeneratedImage = minorityClass + "/generated/" + std::to_string(totalImage[eachClass]+i) + ".png";
				GenerateImage(images, nameGeneratedImage, totalImage[eachClass], generationType);
			}
			rebalanceTotal += rebalance;
			std::cout << rebalance << " images were generated and this is now balanced." << std::endl;
			std::cout << "-------------------------------------------------------" << std::endl;
			images.clear();
		}
	}
	return newDirectory;
}

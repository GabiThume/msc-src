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

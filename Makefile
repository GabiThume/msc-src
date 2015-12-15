CXX = g++
CXXFLAGS = -g -std=c++11 -Wall -I include
FIND = find -name
RM = rm -rf
OPENCV = -I /usr/include/opencv
OPENCVCONF = `pkg-config opencv --libs`
LIB_DIR = lib
VLFEAT = -I $(LIB_DIR)/vlfeat
GMR = $(LIB_DIR)/Saliency/GMRsaliency.cpp
SLIC = $(LIB_DIR)/SLIC/SLIC.cpp
INCLUDES =  $(OPENCV) $(OPENCVCONF) $(VLFEAT) $(GMR) $(SLIC)
SRC_DIR = src
CLASSIFICATION = $(SRC_DIR)/classification/classifier.cpp
DESCRIPTION = $(SRC_DIR)/description/description.cpp
PREPROCESSING = $(SRC_DIR)/preprocessing
QUANTIZATION = $(SRC_DIR)/quantization/quantization.cpp
UTILS = $(SRC_DIR)/utils
EXAMPLES = examples
OBJS = data.o description.o quantization.o classifier.o smote.o rebalance.o artificialGeneration.o

# all: clean descriptorTest artificialGenerationTest rebalanceTest smoteTest rebalanceTwoClasses rebalanceMultiClasses staticRebalance classifierTest cleanLink

all: clean rebalanceTwoClasses

# run: run-desbalanced
#
# run-desbalanced:
# 	./bin/rebalanceTest Desbalanced/original/ features/ 0 # Replication
# 	./bin/rebalanceTest Desbalanced/original/ features/ 1 # All operations
# 	./bin/rebalanceTest Desbalanced/original/ features/ 2 # Blur
# 	./bin/rebalanceTest Desbalanced/original/ features/ 3 # Noise
# 	./bin/rebalanceTest Desbalanced/original/ features/ 4 # Blending
# 	./bin/rebalanceTest Desbalanced/original/ features/ 5 # Unsharp masking
# 	./bin/rebalanceTest Desbalanced/original/ features/ 6 # Composition wuth 4
# 	./bin/rebalanceTest Desbalanced/original/ features/ 7 # Composition with 16
# 	./bin/rebalanceTest Desbalanced/original/ features/ 8 # Threshold combination
# 	./bin/rebalanceTest Desbalanced/original/ features/ 9 # Saliency
#
# plot:
# 	python plot.py
# 	python statistics.py

data.o:
	$(CXX) -c $(CXXFLAGS) $(UTILS)/data.cpp $(OPENCV)

quantization.o:
	$(CXX) -c $(CXXFLAGS) $(QUANTIZATION) $(OPENCV)

description.o:
	$(CXX) -c $(CXXFLAGS) $(DESCRIPTION) $(OPENCV) $(VLFEAT)

classifier.o:
	$(CXX) -c $(CXXFLAGS) $(CLASSIFICATION) $(OPENCV) $(VLFEAT)

smote.o:
	$(CXX) -c $(CXXFLAGS) $(PREPROCESSING)/smote.cpp $(OPENCV) $(VLFEAT)

rebalance.o:
	$(CXX) -c $(CXXFLAGS) $(UTILS)/rebalance.cpp $(OPENCV) $(VLFEAT)

artificialGeneration.o:
	$(CXX) -c $(CXXFLAGS) $(PREPROCESSING)/artificialGeneration.cpp $(OPENCV) $(VLFEAT) $(GMR) $(SLIC)

descriptorTest: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/descriptor $(EXAMPLES)/descriptorTest.cpp $(INCLUDES)

smoteTest: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/smoteTest $(EXAMPLES)/smoteTest.cpp $(INCLUDES)

artificialGenerationTest: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/artificialGenerationTest $(EXAMPLES)/artificialGenerationTest.cpp $(INCLUDES)

rebalanceTest: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/rebalanceTest $(EXAMPLES)/rebalanceTest.cpp $(INCLUDES)

rebalanceTwoClasses: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/rebalanceTwoClasses $(EXAMPLES)/rebalanceTwoClasses.cpp $(INCLUDES)

rebalanceMultiClasses: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/rebalanceMultiClasses $(EXAMPLES)/rebalanceMultiClasses.cpp $(INCLUDES)

staticRebalance: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/staticRebalance $(EXAMPLES)/staticRebalance.cpp $(INCLUDES)

classifierTest: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/classify $(EXAMPLES)/classifierTest.cpp $(INCLUDES)

clean:
	$(RM) ./*.o
	$(RM) ./*.o
	$(RM) ./bin/*

cleanLink:
	$(RM) ./*.o

OPENCV = -I /usr/include/opencv
OPENCVCONF = `pkg-config opencv --libs`
VLFEAT = -I lib/vlfeat
GMR = lib/Saliency/GMRsaliency.cpp
SLIC = lib/SLIC/SLIC.cpp
FIND=find -name
BIN = $(FIND) "*.o"
BINPATH=`bin/`
FLAGS = -std=c++11 -Wall
RM=rm -rf
CLASSIFICATION = source/classification/
DESCRIPTION = source/description/
PREPROCESSING = source/preprocessing/
QUANTIZATION = source/quantization/
UTILS = source/utils/
EXAMPLES = examples/
# all: clean funcoesAux quantization descritores funcoesArquivo merge_datasets classifier dimensionReduction smote smoteTest artificialGeneration artificialGenerationTest rebalanceTest rebalanceTwoClasses rebalanceMultiClasses
# 	@g++ -Wall quantization.o descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor $(OPENCV) $(OPENCVCONF) $(VLFEAT)

funcoesAux:
	@g++ $(FLAGS) -c -g $(UTILS)funcoesAux.cpp $(OPENCV)

quantization:
	@g++ $(FLAGS) -c -g $(QUANTIZATION)quantization.cpp $(OPENCV)

descritores:
	@g++ $(FLAGS) -c -g $(DESCRIPTION)descritores.cpp $(OPENCV) $(VLFEAT)

funcoesArquivo:
	@g++ $(FLAGS) -c -g $(UTILS)funcoesArquivo.cpp $(OPENCV) $(VLFEAT)

classifier:
	@g++ $(FLAGS) -c -g $(CLASSIFICATION)classifier.cpp $(OPENCV) $(VLFEAT)

smote:
	@g++ $(FLAGS) -c -g $(PREPROCESSING)smote.cpp $(OPENCV) $(VLFEAT)

smoteTest: smoteTest.cpp
	@g++ $(FLAGS) $(BIN) $(PREPROCESSING)smoteTest.cpp -o smoteTest $(OPENCV) $(OPENCVCONF) $(VLFEAT)

artificialGeneration:
	@g++ $(FLAGS) -c -g $(GMR) $(SLIC) $(PREPROCESSING)artificialGeneration.cpp -std=c++0x $(OPENCV) $(VLFEAT)

artificialGenerationTest: $(EXAMPLES)artificialGenerationTest.cpp
	@g++ $(FLAGS) $(BIN) $(EXAMPLES)artificialGenerationTest.cpp -o artificialGenerationTest $(OPENCV) $(OPENCVCONF) $(VLFEAT)

rebalanceTest: $(EXAMPLES)rebalanceTest.cpp
	@g++ $(BIN) $(EXAMPLES)rebalanceTest.cpp -o rebalanceTest $(OPENCV) $(OPENCVCONF) $(VLFEAT)

rebalanceTwoClasses: $(EXAMPLES)rebalanceTwoClasses.cpp
	@g++ $(FLAGS) $(BIN) $(EXAMPLES)rebalanceTwoClasses.cpp -o rebalanceTwoClasses $(OPENCV) $(OPENCVCONF) $(VLFEAT)

rebalanceMultiClasses: $(EXAMPLES)rebalanceMultiClasses.cpp
	@g++ $(FLAGS) $(BIN) $(EXAMPLES)rebalanceMultiClasses.cpp -o rebalanceMultiClasses $(OPENCV) $(OPENCVCONF) $(VLFEAT)

staticRebalance: $(EXAMPLES)staticRebalance.cpp
	@g++ $(FLAGS) $(BIN) $(EXAMPLES)staticRebalance.cpp -o staticRebalance $(OPENCV) $(OPENCVCONF) $(VLFEAT)

clean:
	$(FIND) "*~" | xargs $(RM)
	$(FIND) "*.o" | xargs $(RM)
# $(RM) staticRebalance mainDescritor smoteTest artificialGenerationTest rebalanceTest rebalanceMultiClasses rebalanceTwoClasses

# run: run-desbalanced
#
# run-desbalanced:
# 	./rebalanceTest Desbalanced/original/ features/ 0 # Replication
# 	./rebalanceTest Desbalanced/original/ features/ 1 # All operations
# 	./rebalanceTest Desbalanced/original/ features/ 2 # Blur
# 	./rebalanceTest Desbalanced/original/ features/ 3 # Noise
# 	./rebalanceTest Desbalanced/original/ features/ 4 # Blending
# 	./rebalanceTest Desbalanced/original/ features/ 5 # Unsharp masking
# 	./rebalanceTest Desbalanced/original/ features/ 6 # Composition wuth 4
# 	./rebalanceTest Desbalanced/original/ features/ 7 # Composition with 16
# 	./rebalanceTest Desbalanced/original/ features/ 8 # Threshold combination
# 	./rebalanceTest Desbalanced/original/ features/ 9 # Saliency
#
# plot:
# 	python plot.py
# 	python statistics.py
#
# run-dimensionReduction:
# 	./dimensionReduction

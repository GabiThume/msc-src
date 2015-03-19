OPENCV = -I /usr/include/opencv
OPENCVCONF = `pkg-config opencv --libs`
FIND=find -name
RM=rm -rf

all: clean descritores funcoesAux funcoesArquivo merge_datasets classifier dimensionReduction smote smoteTest artificialGeneration artificialGenerationTest rebalanceTest
	@g++ -Wall descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor $(OPENCV) $(OPENCVCONF)

debug: descritores funcoesAux funcoesArquivo
	@g++ -Wall -g descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor $(OPENCV)

teste: descritores funcoesAux
	@g++ -Wall -g descritores.o funcoesAux.o teste.cpp -o teste $(OPENCV)
	
descritores:
	@g++ -Wall -c -g descritores.cpp $(OPENCV)

funcoesAux:
	@g++ -Wall -c -g funcoesAux.cpp $(OPENCV)
	
funcoesArquivo:
	@g++ -Wall -c -g funcoesArquivo.cpp $(OPENCV)

dimensionReduction: dimensionReduction.cpp
	@g++ -Wall -o dimensionReduction descritores.o funcoesAux.o funcoesArquivo.o classifier.o dimensionReduction.cpp $(OPENCV) $(OPENCVCONF)
	
merge_datasets: mergeDataSets.cpp	
	@g++ -Wall -o mergeDataSets mergeDataSets.cpp

classifier:
	@g++ -Wall -c -g classifier.cpp $(OPENCV)

smote:
	@g++ -Wall -c -g smote.cpp $(OPENCV)

smoteTest: smoteTest.cpp
	@g++ -Wall descritores.o funcoesAux.o funcoesArquivo.o classifier.o smote.o smoteTest.cpp -o smoteTest $(OPENCV) $(OPENCVCONF)

artificialGeneration:
	@g++ -Wall -c -g artificialGeneration.cpp -std=c++0x $(OPENCV)

artificialGenerationTest: artificialGenerationTest.cpp
	@g++ -Wall artificialGeneration.o descritores.o funcoesAux.o funcoesArquivo.o classifier.o artificialGenerationTest.cpp -o artificialGenerationTest $(OPENCV) $(OPENCVCONF)

rebalanceTest: rebalanceTest.cpp
	@g++ -Wall artificialGeneration.o descritores.o funcoesAux.o funcoesArquivo.o classifier.o smote.o rebalanceTest.cpp -o rebalanceTest $(OPENCV) $(OPENCVCONF)

clean:
	$(FIND) "*~" | xargs $(RM)
	$(RM) teste mainDescritor dimensionReduction mergeDataSets smoteTest artificialGenerationTest rebalanceTest

run: run-desbalanced

run-desbalanced:
	./rebalanceTest Desbalanced/original/ features/ 0 # All operations
	./rebalanceTest Desbalanced/original/ features/ 1 # Blur
	./rebalanceTest Desbalanced/original/ features/ 2 # Noise
	./rebalanceTest Desbalanced/original/ features/ 3 # Blending
	./rebalanceTest Desbalanced/original/ features/ 4 # Unsharp masking
	./rebalanceTest Desbalanced/original/ features/ 5 # Composition
	./rebalanceTest Desbalanced/original/ features/ -1 # Replication

plot:
	python plot.py
	python statistics.py

run-dimensionReduction:
	./dimensionReduction


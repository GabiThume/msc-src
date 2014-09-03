all: descritores funcoesAux funcoesArquivo dimensionReduction merge_datasets classifier smote smoteTest rebalance rebalanceTest
	@g++ descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor -I /usr/include/opencv `pkg-config opencv --libs`

debug: descritores funcoesAux funcoesArquivo
	@g++ -g descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor -I /usr/include/opencv `pkg-config opencv --libs`

teste: descritores funcoesAux
	@g++ -g descritores.o funcoesAux.o teste.cpp -o teste -I /usr/include/opencv `pkg-config opencv --libs`
	
descritores:
	@g++ -c -g descritores.cpp -I /usr/include/opencv `pkg-config opencv --libs`

funcoesAux:
	@g++ -c -g funcoesAux.cpp -I /usr/include/opencv `pkg-config opencv --libs`
	
funcoesArquivo:
	@g++ -c -g funcoesArquivo.cpp -I /usr/include/opencv `pkg-config opencv --libs`

dimensionReduction: dimensionReduction.cpp
	@g++ -o dimensionReduction dimensionReduction.cpp -I /usr/include/opencv `pkg-config opencv --libs`
	
merge_datasets: mergeDataSets.cpp	
	@g++ -o mergeDataSets mergeDataSets.cpp

smoteTest: smoteTest.cpp
	@g++ -Wall descritores.o funcoesAux.o funcoesArquivo.o classifier.o smote.o smoteTest.cpp -o smoteTest -I /usr/include/opencv `pkg-config opencv --libs`

smote:
	@g++ -Wall -c -g smote.cpp -I /usr/include/opencv `pkg-config opencv --libs`

classifier:
	@g++ -Wall -c -g classifier.cpp -I /usr/include/opencv `pkg-config opencv --libs`

rebalanceTest: rebalanceTest.cpp
	@g++ -Wall rebalance.o descritores.o funcoesAux.o funcoesArquivo.o classifier.o rebalanceTest.cpp -o rebalanceTest -I /usr/include/opencv `pkg-config opencv --libs`

rebalance:
	@g++ -Wall -c -g rebalance.cpp -std=c++0x -I /usr/include/opencv `pkg-config opencv --libs`

clean:
	rm *.o *.*~ teste *~ mainDescritor dimensionReduction mergeDataSets smoteTest rebalanceTest



#include "funcoesArquivo.h"

int main(int argc, char *argv[]){

    int descMethod, colors, normalization, quantMethod, deleteNull, numParameters;
    vector<int> params;
    double resize;
    string id = "", databaseDir = "", descMethodDir = "";

    if (argc < 8){
        cout << "O programa espera: <pasta> <pasta_descritor> <descritor> <quantidade de cores para quantizar a imagem> ";
        cout << "fator de redimensionamento> <normalizacao> <metodo quantizacao> <distancias ACC | limiar CCV> ";
        cout << "id para arquivo de saida, caso necessário>" << endl << endl;
        cout << "\tDescritores:\t1 - BIC\t2 - GCH\t3 - CCV\t4 - Haralick\t5 - AutoCorrelograma (ACC)\t6 - LBP\t7 - HOG\t8 - Contour Extraction\t9 - Fisher Vectors" << endl;
        cout << "\tCores:\t8, 16, 32, 64 ou 256\n";
        cout << "\tRedimensionamento:\tpositivo, com máximo = 1 (1 = 100%)\n";
        cout << "\tNormalizacao\t0 (sem normalizacao) 1 (entre 0 e 1), 2 (0 a 255)\n";
        //cout << " - Descartar colunas nulas (atributos nulos) - 1 descartar, 0 nao descartar\n";
        cout << "\tSequencia de distancias para ACC ou limiar do CCV\n";
        exit(0);
    }

    if (argc == 9){
        id = argv[8];
    }

	databaseDir = argv[1];
	descMethodDir = argv[2];

	descMethod = atoi(argv[3]);
	if(descMethod < 1 || descMethod > 10) {
	    cout << "This descriptor method does not exist.\n";
        return -1;
	}

	colors = atoi(argv[4]);
	if(colors != 8 && colors != 16 && colors != 32 && colors != 64 && colors != 128 && colors != 256){
	    cout << "The number of colors must be 8, 16, 32, 64, 128 or 256." << endl;
        return -1;
	}

	resize = atof(argv[5]);
	if(resize <= 0 || resize > 1){
	    cout << "The resize factor must be between 0 and 1." << endl;
        return -1;
	}

	normalization = atoi(argv[6]);
	if(normalization < 0 || normalization > 2){
	    cout << "Invalid normalization (use 0, 1 or 2)." << endl;
        return -1;
	}

	quantMethod = atoi(argv[7]);

 	// deleteNull = atoi(argv[6]); - delete null columns
	deleteNull = 0;
	if(deleteNull != 0 && deleteNull != 1){
	    cout << "Wrong delete option (0, 1)." << endl;
        return -1;
	}

	numParameters = (argc-8);
	if (descMethod == 3 || descMethod == 5) {
      for (int i = 0; i < numParameters; i++){
          params.push_back(atoi(argv[6+i]));
      }
	}

	descriptor(databaseDir, descMethodDir, descMethod, colors, resize, normalization, params, deleteNull, quantMethod, id);
	return 0;
}

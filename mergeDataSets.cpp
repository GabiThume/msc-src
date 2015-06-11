/*
    Run: ./<executavel> <arquivo_saida> <numero_datasets> <dataset_1> ... <dataset_N>
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

void error(){
    printf("Entrada incorreta!!! Use:\n");
    printf("\t ./<executavel> <arquivo_saida> <numero_datasets> <dataset_1> ... <dataset_N>");
}

int main (int argc, char **argv){

    int nFiles, i, j, k, *nAttrsDataSets, nInstances, nAttrs, nInstancesAux, nClassesAux, nAttrsAux, label, nInst;
    double attrVal;
    FILE **files, *fout;

    if(argc < 5){
        error();
        exit(-1);
    }

    nFiles = atoi(argv[2]);
    if(argc != nFiles + 3){
        error();
        exit(-1);
    }

    files = (FILE**)malloc(nFiles * sizeof(FILE*));
    nAttrsDataSets = (int*)malloc(nFiles * sizeof(int));
    nAttrs = 0;

    for(i = 0; i < nFiles; ++i){
        files[i] = fopen(argv[i + 3], "r");
        fscanf(files[i], "%d %d %d", &nInstancesAux, &nClassesAux, &nAttrsAux);
        nAttrsDataSets[i] = nAttrsAux;
        nAttrs += nAttrsAux;
    }
    nInstances = nInstancesAux;
    fout = fopen(argv[1], "w");
    for(i = 0; i < nInstances; ++i){
        //if(i == 0)
        //  fprintf(fout, "%d\t%d\t%d\n", nInstances, nClassesAux, nAttrs);

        for(j = 0; j < nFiles; ++j){
            fscanf(files[j], "%d\t%d", &nInst, &label);
            if(j == 0){
               fprintf(fout, "%d\t%d\t", nInst, label);
            }

            for(k = 0; k < nAttrsDataSets[j]; ++k){
                fscanf(files[j], "%lf", &attrVal);
                fprintf(fout, " %lf", attrVal);
            }
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
    for(i = 0; i < nFiles; ++i){
        fclose(files[i]);
    }
    free(nAttrsDataSets);
    free(files);
    return 0;
}
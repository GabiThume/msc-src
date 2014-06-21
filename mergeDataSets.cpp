#include <stdio.h>
#include <stdlib.h>

// Entrada:
//  ./<executavel> <arquivo_saida> <numero_datasets> <dataset_1> ... <dataset_N>

void error(){
  printf("Entrada incorreta!!! Use:\n");
  printf("\t ./<executavel> <arquivo_saida> <numero_datasets> <dataset_1> ... <dataset_N>");
}

int main (int argc, char **argv){
  char *outName;
  int nFiles, i, j, k, *nAttrsDataSets, nInstances, nClasses, nAttrs, nInstancesAux, nClassesAux, nAttrsAux, label, nInst;
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
  nClasses = nClassesAux;
  
  fout = fopen(argv[1], "w");
  
  for(i = 0; i < nInstances; ++i){
    if(i == 0){
      fprintf(fout, "%d %d %d\n", nInstances, nClasses, nAttrs);
    }
    
    for(j = 0; j < nFiles; ++j){
      fscanf(files[j], "%d %d", &nInst, &label);
      if(j == 0){
        fprintf(fout, "%d %d", nInst, label);
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

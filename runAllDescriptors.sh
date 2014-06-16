#!/bin/bash
#
# ./mainDescritor <base_imagens> <descritor> <redimensionamento> <normalizacao> <metodo_quantizacao> <parametros>
#
# Descritor: 1-BIC 2-GCH 3-CCV 4-Haralick 5-AutoCorrelograma(ACC)
# Cores: 8, 16, 32, 64 ou 256
# Redimensionamento: positivo, com máximo = 1 (1 = 100%)
# Normalizacao: 0 (sem normalizacao) 1 (entre 0 e 1), 2 (0 a 255)
# Metodo de quantizacao: 1-Intensity 2-Gleam 3-Luminance 4-MSB
# Parametros: Sequencia de distancias para ACC ou limiar do CCV

for i in {1..5}
do
    for j in {1..4}
    do
      # Se o descritor for CCV, passa o parâmetro de threshold
      if [ $i == 3 ] ; then
          ./mainDescritor BaseImagens $i 256 1 1 $j 25
      # Se o descritor for ACC, passa como parâmetros as distâncias
      elif [ $i == 5 ] ; then
          ./mainDescritor BaseImagens $i 256 1 1 $j 1 3 5 7
      else
          ./mainDescritor BaseImagens $i 256 1 1 $j
      fi
    done
done

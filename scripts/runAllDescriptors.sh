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

cor[0]=256
cor[1]=128
cor[2]=64
cor[3]=32
cor[4]=16
cor[5]=8

for cores in {0..5}
do
	for descriptor in {1..5}
	do
		for quantization in {1..4}
		do
		# Se o descritor for CCV, passa o parâmetro de threshold
		if [ $descriptor == 3 ] ; then
		  ./mainDescritor BaseCorel caracteristicas_corel/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 25
		  ./mainDescritor BaseCaltech caracteristicas_caltech600/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 25
		  ./mainDescritor BaseTropical caracteristicas_tropical_fruits1400/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 25
		# Se o descritor for ACC, passa como parâmetros as distâncias
		elif [ $descriptor == 5 ] ; then
		  ./mainDescritor BaseCorel caracteristicas_corel/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 1 3 5 7
		  ./mainDescritor BaseCaltech caracteristicas_caltech600/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 1 3 5 7
		  ./mainDescritor BaseTropical caracteristicas_tropical_fruits1400/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization 1 3 5 7
		else
		  ./mainDescritor BaseCorel caracteristicas_corel/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization
		  ./mainDescritor BaseCaltech caracteristicas_caltech600/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization
		  ./mainDescritor BaseTropical caracteristicas_tropical_fruits1400/${cor[cores]} $descriptor ${cor[cores]} 1 1 $quantization
		fi
		done
		done
done


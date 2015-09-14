#!/bin/bash
#

cor[0]=8
cor[1]=16
cor[2]=32
cor[3]=64
cor[4]=128
cor[5]=256

dim[0]=10
dim[1]=30
dim[2]=50
dim[3]=100

mkdir build
cd build
cmake ..
make

for cores in {0..5}
do
	for dimension in {0..3}
	do
	  echo ${cor[cores]} ${dim[dimension]}
	  ./lpp ../../caracteristicas_corel/${cor[cores]}/all/ ${dim[dimension]} 
	  ./lpp ../../caracteristicas_caltech600/${cor[cores]}/all/ ${dim[dimension]} 
	  ./lpp ../../caracteristicas_tropical_fruits1400/${cor[cores]}/all/ ${dim[dimension]} 
	  # ./lpp ../../caracteristicas_corel/${cor[cores]}/all/ ${dim[dimension]} > ../../analysis/Corel/LPP_ALL_${dim[dimension]}_${cor[cores]}.txt
	  #. /lpp ../../caracteristicas_caltech600/${cor[cores]}/all/ ${dim[dimension]} > ../../analysis/Caltech/LPP_ALL_${dim[dimension]}_${cor[cores]}.txt
	  # ./lpp ../../caracteristicas_tropical_fruits1400/${cor[cores]}/all/ ${dim[dimension]} > ../../analysis/Tropical-Fruits/LPP_ALL_${dim[dimension]}_${cor[cores]}.txt
	done
done


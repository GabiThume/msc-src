#!/bin/bash
#
# ./mergeDataSets <novo_arquivo> <quantidade_arquivos> <nome_dos_arquivos>
#
# Mude para .txt caso precise, e descomente a primeira linha no arquivo mergeDataSets.cpp


cor[0]=256
cor[1]=128
cor[2]=64
cor[3]=32
cor[4]=16
cor[5]=8

base[0]="caracteristicas_corel"
base[1]="caracteristicas_tropical_fruits1400"
base[2]="caracteristicas_caltech600"

baseimg[0]="corel"
baseimg[1]="tropical_fruits1400"
baseimg[2]="caltech600"

for i in {0..5}
do
	for j in {0..2}
	do
		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Gleam))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Gleam_${cor[i]}c_4d_100r.csv 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Intensity))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Intensity_${cor[i]}c_4d_100r.csv 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Luminance))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Luminance_${cor[i]}c_4d_100r.csv 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep MSB))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_MSB_${cor[i]}c_4d_100r.csv 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}
	done
done

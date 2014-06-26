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

for i in {1..5}
do
	for j in {0..2}
	do
		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Gleam))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Gleam_256c_4d_100r.txt 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Intensity))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Intensity_256c_4d_100r.txt 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep Luminance))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_Luminance_256c_4d_100r.txt 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}

		arq=($(find ${base[j]}/${cor[i]} -maxdepth 1 | grep MSB))
		./mergeDataSets ${base[j]}/${cor[i]}/all/${baseimg[j]}_ALL_MSB_256c_4d_100r.txt 5 ${arq[0]} ${arq[1]} ${arq[2]} ${arq[3]} ${arq[4]}
	done
done

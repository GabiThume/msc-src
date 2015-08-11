# renomeia diretorios de A a N
# no terminal digite ./renamefolders.sh A
# exemplo: ./renamefolders.sh 1
# -> vai renomear os diretorios de 1 ate o numero de diretorios que houverem no local

# i=$1 # inicio de contagem passado por parametro ($1)
# for file in *; do
#       if [ -d $file ]; then
#         j=$( printf "%d" "$i" )
#         mv "$file" "$j"
#         i=$((i + 1))
#       fi
# done

# renomeia imagens jpg de A a N dentro de cada subdiretorio
# se a base for em .png ou outro formato deve ser feita a troca da string no script
# exemplo: ./renameimgs.sh 0
# -> renomeia imagens de 0 ate o total de imagens que houver em cada subpasta
for file in *; do
      if [ -d $file ]; then
         cd $file;
	 i=$1
	 for file in *.jpg
	 do
	    j=$( printf "%d" "$i" )
	    mv "$file" "$j.jpg"
	    i=$((i + 1))
	 done
	 cd ..
      fi
done


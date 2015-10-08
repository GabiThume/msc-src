#!/bin/bash
# new=0
# ext=
#
# for i in *
# do
#         ext="${i#*.}"
# 	echo ext
#         #mv "$i" "$new.$ext"
#         ((new++))
#
# done

# zero filling
mogrify -format png *.jpg; rm *.jpg;

i=$1
for file in *.png
do
        j=$( printf "%d" "$i" )
        if [ -f "$j.png" ]
        then
          echo "File $j.png exists"
        else
          echo "mv $file $j.png"
          mv "$file" "$j.png"
        fi
        convert "$j.png" -resize 256x256 "$j.png"
        i=$((i + 1))
done

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
i=$1
for file in *.jpg
do
        j=$( printf "%d" "$i" )
        mv "$file" "$j.jpg"
        i=$((i + 1))
done

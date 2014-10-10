cd $1
for d in *; do
    a=0
    for i in $d/*.jpg; do
      new=$(printf "$d/x%d.jpg" ${a}) #04 pad to length of 4
      mv ${i} ${new}
      let a=a+1
    done
    a=0
    for i in $d/*.jpg; do
      new=$(printf "$d/%d.jpg" ${a}) #04 pad to length of 4
      mv ${i} ${new}
      let a=a+1
    done
done
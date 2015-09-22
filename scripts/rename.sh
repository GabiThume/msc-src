cd $1
# for d in *; do
#     a=0
#     for i in $d/*.jpg; do
#       new=$(printf "$d/x%d.jpg" ${a})
#       mv ${i} ${new}
#       let a=a+1
#     done
#     a=0
#     for i in $d/*.jpg; do
#       new=$(printf "$d/%d.jpg" ${a})
#       mv ${i} ${new}
#       let a=a+1
#     done
# done
for d in *; do
    a=0
    for i in $d/*.png; do
      new=$(printf "$d/x%d.png" ${a})
      mv ${i} ${new}
      let a=a+1
    done
    a=0
    for i in $d/*.png; do
      new=$(printf "$d/%d.png" ${a})
      mv ${i} ${new}
      let a=a+1
    done
done

dir=$1
out=$2

for file in $dir/*
do
        echo $file
        base=$(basename $file .txt)
	v2=${file: -3}
	echo $base
	echo $v2
	echo $out
	if [ "$v2" = "txt" ]; then
        	echo ../tr_output/$base.err
       		sbatch -o ../tr_output/$base.out -e ../tr_output/$base.err h_train_lang_adapt.slurm $file $base $out
	fi 
   #whatever you need with "$file"
done

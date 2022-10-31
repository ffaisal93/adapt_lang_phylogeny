dir=$1
out=$2
fname=$3
eval=$4

for file in $dir/*
do
        echo $file
        base=$(basename $file .txt)
	v2=${file: -3}
	echo $base
	echo $v2
	echo $out

	if [ "$v2" != "txt" ] && [ "$out" = "udp_mlm" ] && [ "$file" != ".DS_Store" ] && [ "$eval" != "train" ]; then
        	echo ../tr_output/$base.err, $file $base $out $fname $eval

        if [ "$v2" != "txt" ] && [ "$out" = "pos_mlm" ] && [ "$file" != ".DS_Store" ] && [ "$eval" != "train" ]; then
        	echo ../tr_output/$base.err, $file $base $out $fname $eval
       		sbatch -o ../tr_output/$base.out -e ../tr_output/$base.err train_task_mlm.slurm $file $base $out $fname $eval
       		
       	elif [ "$v2" != "txt" ] && [ "$out" = "udp_mlm" ] && [ "$file" != ".DS_Store" ] && [ "$eval" != "udp_eval" ]; then
       		echo "udp eval----------------------"
        	echo ../tr_output/$base.err, $file, $out, $fname
       		sbatch -o ../tr_output/$base.out -e ../tr_output/$base.err train_task_mlm.slurm $file $base $out $fname $eval
	elif [ "$v2" = "txt" ]; then
        	echo ../tr_output/$base.err
       		sbatch -o ../tr_output/$base.out -e ../tr_output/$base.err h_train_lang_adapt.slurm $file $base $out
	fi 
   #whatever you need with "$file"
done



#./all_train.sh ../adapters/mbert/uralic-mlm udp_mlm uralic
#./all_train.sh ../adapters/mbert/germanic-mlm udp_mlm germanic
#./all_train.sh ../adapters/mbert/tupian-mlm udp_mlm tupian


##udp-mlm-eval
#./all_train.sh ../experiments/mbert/udp_mlm/tupian udp_mlm tupian udp_eval
#./all_train.sh ../experiments/mbert/udp_mlm/germanic udp_mlm germanic_up udp_eval
#./all_train.sh ../experiments/mbert/udp_mlm/uralic udp_mlm uralic_up udp_eval

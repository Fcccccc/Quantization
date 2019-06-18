cd ..
for i in `seq $1 0.1 $2`
do
	CUDA_VISIBLE_DEVICES=$3 ./load_and_run.py Resnet34 -p $i -q $4
done

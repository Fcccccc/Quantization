cd ..
for i in `seq $1 0.05 $2`
do
	CUDA_VISIBLE_DEVICES=$3 ./load_and_run.py Resnet18 -p $i -q $4
done

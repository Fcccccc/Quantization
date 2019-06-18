cd ..
for i in `seq 0.6 0.05 0.95`
do
	CUDA_VISIBLE_DEVICES=$1 ./load_and_run.py Vgg16 -p $i -q $2
done

cd ..
for i in `seq 0.6 0.05 0.95`
do
	CUDA_VISIBLE_DEVICES=0 ./load_and_run.py Alexnet -p $i -q $1
done

#for i in `seq 0.6 0.05 0.65`
#do
	#CUDA_VISIBLE_DEVICES=0 ./load_and_run.py Resnet18 -p $i -q $1
#done

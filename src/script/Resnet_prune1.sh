cd ..
for i in `seq 0.85 0.05 0.95`
do
	CUDA_VISIBLE_DEVICES=3 ./load_and_run.py Resnet18 -p $i -q 4
done

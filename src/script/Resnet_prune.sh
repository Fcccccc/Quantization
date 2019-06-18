cd ..

gpubegin=$1
quant=$2

test_gpu(){
	num=$((($1 + 4) % 8))
	until [ -n "`nvidia-smi -i $num | tail -n 2 | head -n 1 | grep "No running processes found"`" ]
	do
		echo "not really"
		sleep 60
	done
}


run(){
	for i in `seq $2 0.05 ${2}5`
	do
		test_gpu $1
		echo "CUDA_VISIBLE_DEVICES=$1 ./load_and_run.py Resnet34 -q $quant -p $i >/dev/null 2>&1"
		#CUDA_VISIBLE_DEVICES=$1 ./load_and_run.py Resnet34 -q $quant -p $i > /dev/null 2>&1 &
		sleep 180
	done
}


for i in `seq 0.6 0.1 0.9`
do
	run $gpubegin $i &
	gpubegin=$(($gpubegin + 1)) 
done
wait


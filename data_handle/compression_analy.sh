modelname=$1
method=$2
arg=""
for file in `find compression_data -type f | grep $modelname | grep mask | grep quant`
do
	arg=$arg" "$file
done
#echo "`find compression_data -type f | grep $modelname | grep mask | grep quant`"
#echo $arg
python3 ./compression.py $2 $arg

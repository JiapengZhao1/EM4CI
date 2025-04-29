#!/bin/bash

#$1 is the name of your model $2 is the number of samples
#models are assumed to be in "models_xdsl" folder
ED=(2 4 6 8 10 12 14 16 18 20 22 24)
#ED=(2)
numS=$2
model=$1
edCount=0

mkdir -p $model
mkdir -p $model"/"${numS}
mkdir -p "learned_models/"$model
mkdir -p "learned_models/"$model"/"${numS}
BIC=100000000000

begin=$(date +%s.%N)
learnedModelIndex=0
for ed in ${ED[@]}
do
	echo $ed
	
	
	emModel="em_"${model}"_ED"${ed}
	mkdir -p $model"/"${numS}"/"${emModel}
	LL=()
	time=()
	count=0
	maxLL=0
	maxIndex=0
	start=$(gdate +%s.%N)
	for i in {0..9}
	do
		
		 ./learn.out models_xdsl/${model}.xdsl models_xdsl/${emModel}_${i}.xdsl data/${numS}/${model}.csv ${numS}
	
		LL+=($( tail -1 learned_models/${model}/${numS}/${emModel}/LL.csv))
		time+=($( tail -1 learned_models/${model}/${numS}/${emModel}/timesLearn.csv))
#	count=$count+1
	done
	stop=$(gdate +%s.%N)
	elapsed=$(bc -l <<< "$stop - $start")
	#maxLL=${LL[0]}
	maxLL=-100000000000
	maxIndex=0
	count=0
	echo LL ${LL[0]}
	for i in ${LL[@]}; do
		#echo $i
		#echo $count
		if (( $(echo "$i > $maxLL && $i != 0" | bc -l) )); then
		#if [[ ${LL[$i]} -gt  $maxLL ]]; then
			maxLL=$i
			maxIndex=$count
		fi
		((count+=1))
	done
	echo "the max LL is " ${LL[$maxIndex]}" from run # " $maxIndex " for estimated domain size " $ed
	echo ${LL[$maxIndex]} >> "learned_models/${model}/${numS}/${emModel}/wrapper_LL.csv"
	echo $elapsed >> "learned_models/${model}/${numS}/${emModel}/wrapper_time.csv"
	currentBIC=$( ./bic.sh ${model} ${LL[$maxIndex]} $ed ${numS} )
	echo "the BIC score for the max LL is  " $currentBIC " for estimated domain size " $ed
	#echo ${currentBIC} >> "${model}/${numS}/${emModel}/wrapper_bic.csv"
	if  (( $(echo "$currentBIC < $BIC" | bc -l) )); then
		BIC=$currentBIC
		learnedModelIndex=$maxIndex
		echo "learned model " $learnedModelIndex
	else
		echo -e "\n\n"
		echo "BIC score has started to increase again the final BIC Score is " ${BIC} "corresponding to domain size" ${ED[$edCount-1]} 
		echo "the learned model that produces this score is " ${learnedModelIndex} 
		echo ${currentBIC} >> "learned_models/${model}/${numS}/wrapper_finalbic.csv"
		echo ${learnedModelIndex} >> "learned_models/${model}/${numS}/wrapper_learnedModelNum.csv"
		echo ${ED[$edCount-1]} >> "learned_models/${model}/${numS}/wrapper_domain.csv"
		end=$(gdate +%s.%N)
		total_time=$(bc -l <<< "$stop - $start")
		echo "Total elapsed time is " ${total_time} 
		echo $total_time >> "learned_models/${model}/${numS}/wrapper_time.csv"
		exit 0
	fi

	((edCount+=1))
done

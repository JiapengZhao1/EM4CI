#!/bin/bash

#TD=2
#LD=10
#$1 is the model $2 is the LL and $3 is the estimated domain and $4 is the numS
LL=$2
ed=$3
numS=$4
numU=0
#newFile="BIC.csv"
BASEDIR=$1
model=$BASEDIR

if [ $model == "ex1_TD2_10" ] || [ $model ==  "ex6_TD2_10"  ]; then
	numU=2
fi
if [ $model == "ex2_TD2_10"  ] || [ $model == "ex3_TD2_10" ] || [ $model == "ex7_TD2_10"  ]; then
	numU=3
fi
if [ $model == "ex4_TD2_10"  ] || [ $model == "ex5_TD2_10"  ]  || [ $model == "ex8_TD2_10"  ]; then
	numU=4
fi
if [ $model == "A" ]; then 
	numU=8
fi
if [ $model == "BarleyMain" ]; then 
	numU=6
fi
if [ $model == "Alarm" ]; then
	numU=5
fi
if [ $model == "Win95pts" ]; then 
	numU=17
fi
if [ $model == "25_chain_TD4_10" ]; then 
	numU=12
fi
if [ $model == "65_diamond_TD4_10" ]; then 
	numU=32
fi
if [ $model == "45_cone_cloud_TD4_8" ]; then
	numU=16
fi
if [ $model == "napkin" ]; then
	numU=2
fi
if [ $model == "mediator" ]; then
	numU=3
fi
if [ $model == "planid" ]; then
	numU=3
fi
#echo "numU "
#echo ${numU}
#echo '\n'	
BIC=$(awk  -v U="${numU}" -v S="${numS}" -v k="${ed}" -v l="${LL}" 'BEGIN {BIC=(-2*l + U*(k-1)*log(S)); print BIC }' /dev/null) 
#echo "$BIC" > ${BASEDIR}/${numS}"/wrapper_BIC.csv"
echo "$BIC"

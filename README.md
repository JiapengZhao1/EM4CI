# EM4CI
In order to run the code for this paper, a license to use BayesFusion software package SMILE is required.
EM4CI is written in C++. There is one source code file for the learning phase and one for inference. 
They are named: 
    learn_main.cpp   inf.cpp 


In order to compile the source files, the command is:

    g++ -O3 learn_main.cpp -o learn.out -I./smile -L./smile -lsmile
    g++ -O3 inf.cpp -o inf.out -I./smile -L./smile -lsmile
    
Which will produce the executables:

    learn.out         inf.out

The  learn.out file expects command line arguments of the model file, the em-model file with a domain specified for the unobserved variables, a data csv file containing samples on the observed variables, and the number of samples. An example run is:

    ./learn.out models_xdsl/ex1_TD2_10.xdsl  models_xdsl/em_ex1_TD2_10_ED2_0.xdsl data/100/ex1_TD2_10.csv  100

The  inf.out file expects command line arguments of the model file, the learned model file that will be used to perform inference on, the query variable, Y, in P(Y|do(X)), the do variables(s) X, and the number of samples used in the learning phase. An example run is:

    ./inf.out models_xdsl/ex1_TD2_10.xdsl learned_models/100/em_ex1_TD2_10_ED2_0.xdsl  Y X 100

The resulting log-likelihood, BIC score, time for learning, time for inference, and mad(mean absolute deviation) are output to csv files, LL.csv, BIC.csv, timesLearn.csv, timesInf.csv, and err.csv, respectively. These will all be output to a folder named after the model, and containing a different subfolder per sample size and assumed domain size in the learning phase. For example, if running for assumed domain size 2 and sample size 100, the output will be in folder:

    ex1_TD2_10/100/em_ex1_TD2_10_ED2 
    
The learned model files will be in the folder:

    learned_models/ex1_TD2_10/100 

The bash script  em4ci_wrapper.sh was used to automate the learning process. This script will automatically iterate through increasing latent domain sizes, while running the EM algorithm 10 times for each latent domain size. It will stop when the BIC score stops decreasing, and will output the minimum BIC score with the latent domain size of the final learned model.
To run this script you can pass in the model name and number of samples. For example:

    ./em4ci_wrapper.sh ex1_TD2_10 100

The wrapper assumes all model files are contained in a folder named models_xdsl and data files are contained in a folder called data with subfolders corresponding to the number of samples, like  data/100. 

The learned models files are of the form em_ex1_TD2_10_ED2_0.xdsl  where the number after ED corresponds to the assumed domain size of the latent variables, and the last number corresponds to one of the runs {0,.., 9} that produced that model. You can perform inference on any learned model you like, but the em4ci_wrapper.sh outputs the run that correspond to the highest likelihood models with minimum BIC score, so we suggest using those.


All model files are in XDSL format, for more information see the Bayefusion Documentation

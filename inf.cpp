//g++  -O3 inf.cpp -o inf.out -I./smile -L./smile -lsmile
//usage ./inf.out models_xdsl/model.xdsl learned_models/model_file.xdsl queryVar doVar numS
#define SMILE_NO_V1_COMPATIBILITY

#include "smile.h"
#include "smile_license.h"
#include <iostream>
#include <set>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <ctime>
#include <chrono>
#include <cmath>
//using namespace std;

static void PrintMatrix(DSL_network net, const DSL_Dmatrix &mtx, const DSL_idArray &outcomes, const DSL_intArray &parents);

static int CreateCptNode(DSL_network &net, const char *id, const char *name, 
    std::initializer_list<const char *> outcomes, int xPos, int yPos);
std::string PrintVectorInfo(DSL_network &net, std::vector<std::pair<int,int > > domainVec, std::vector<int> currentCombo );

static void PrintPosteriors(DSL_network &net, int handle);
static void PrintNodeInfo(DSL_network &net, int nodeHandle);
int main(int argc, char *argv[]){

DSL_network net;
DSL_network learnedModel;
DSL_network doX;
DSL_network doX2;
DSL_dataset ds;
std::vector<DSL_datasetMatch> matching;
std::string errMsg;
std::ofstream outTinf;
std::ofstream outTlearn;
std::ofstream outS;
std::ofstream outLL;
//double loglik;
std::string outPath;
std::string modelFile = "";
//std::string emFile = "";
//std::string dataFile = "";
std::string learnedFile="";
//std::size_t domain = 2;
std::string basename;
std::size_t dot=0;//unisigned integer
std::size_t slash=0;
//time_t startTime = 0;
//time_t endTime =0;
//std::chrono startTime; 
//std::chrono endTime;
std::string queryVar;
//std::string doNode;
int numPts=0;
int hypDomain= 0;
int domain;
int doDomain;
double totalError =0;
double avgError =0;
//int doVal=0;
int doDim=1;
int numDo;
bool print = true;
if(argc <5)
{
        std::cout << "not enough command line arguments passed" << std::endl;
}
else{
        modelFile = argv[1];
        learnedFile =  argv[2];
        queryVar = argv[3];
        numDo = argc -5;
	std::cout << "num of do vars " << numDo << std::endl;
}
std::string doArray[numDo];
int argvIn=4;
int j=0;
for( j = 0; j < numDo; j++){
        std::cout << "do vars " <<  argv[argvIn + j] << std::endl;
        doArray[j]=argv[argvIn+j];
        std::cout << "in array " << doArray[j] << std::endl;
}
argvIn = argvIn+j;
std::cout << "index " << argvIn << std::endl;
numPts = std::stoi(argv[argvIn]);


dot = modelFile.find('.');
slash = modelFile.rfind('/');
basename=modelFile.substr(slash+1,dot-(slash+1));
outPath = basename + "/"+ std::to_string(numPts)+"/";
std::cout << "outpath " << outPath << std::endl;
outTlearn.open(outPath+"timesInf.csv", std::ios_base::app);
outS.open(outPath+"err.csv", std::ios_base::app);






//int endoIds[numEndo]={0,0,0,0,0,0,0,0,0};
int res = net.ReadFile(modelFile.c_str());
std::cout << "the model file is : " << modelFile.c_str() << std::endl;
if (DSL_OKAY != res)
{
	return res;
}

res = learnedModel.ReadFile(learnedFile.c_str());
std::cout << "the learned  model file is : " << learnedFile.c_str() << std::endl;
if (DSL_OKAY != res)
{
        return res;
}


doX = DSL_network(net);
int handle = doX.FindNode(queryVar.c_str());



if (handle >= 0)
{
	printf("Handle of my query node is: %d\n", handle);
}
else
{
	std::cout<<"There's no node with ID=myNodeId\n";
}

DSL_node* doNode = doX.GetNode(doArray[0].c_str());
int doHandle = doX.FindNode(doArray[0].c_str());
const DSL_idArray& doOut = *doNode->Def()->GetOutcomeIds();
doDomain= doOut.GetSize();
int weightedSize=doDomain;
if (doHandle >= 0)
{
	printf("Handle of my do node is: %d\n", doHandle);
}
else
{
	std::cout<<"There's no node with ID=myNodeId\n";
}
std::cout << "setting up do variables " << std::endl;
doX.SetTarget(doHandle, true );
if(numDo>1)
{
	for (j=1; j<numDo; j++){
		doHandle = doX.FindNode(doArray[j].c_str());
		const DSL_idArray& doOut = *doNode->Def()->GetOutcomeIds();
		doDomain= doOut.GetSize();
		weightedSize*=doDomain;
		doX.SetTarget(doHandle, true );
	}
	
	doX.EnableJptStore(true);
}

doX.UpdateBeliefs();
std::cout << "weightedsize" << weightedSize << std::endl;
std::cout <<"setting up to compute marginal of do vars" << std::endl;
double doMarginal[weightedSize];
if (numDo>=1){
	DSL_node* n = doX.GetNode(doArray[0].c_str());
	const DSL_Dmatrix& beliefs = *n->Val()->GetMatrix();
	const DSL_idArray& outcomes = *n->Def()->GetOutcomeIds();
	const DSL_nodeVal* values = n->Val();
	std::cout << "beliefs size " << beliefs.GetSize() << std::endl;
	std::cout << "do var" << doArray[0].c_str() << std::endl;
	for (int i =0; i < outcomes.GetSize(); i++){
	        doMarginal[i] = beliefs[i];
       		std::cout << "marginal prob of " << outcomes[i] << " " << beliefs[i] << std::endl; 	       
	}
}
else
{
	//	doX.EnableJptStore(true);
	//	To store the JPTs
	std::vector<	std::pair<std::vector<int>,const DSL_Dmatrix *>> vecJPTs;
	int err= doX.GetJpts(doHandle,vecJPTs);
	std::cout << "vars in joint table " << std::endl;
	for( int item : vecJPTs[0].first ){
		std::cout << item << " ";
	}
	std::cout << std::endl;
	if (vecJPTs[0].second != NULL) {
        	const DSL_Dmatrix* matrix = vecJPTs[0].second;
        	std::cout << "joint table:" << std::endl;
        	//PrintMatrix(doX, &matrix, 
		int dimCount = matrix->GetNumberOfDimensions();
		DSL_intArray coords(dimCount);
		//coords.FillWith(0);
		/*
		for (int elemIdx = 0; elemIdx < mtx.GetSize(); elemIdx++)
		{
			//const char *outcome = outcomes[coords[dimCount - 1]];
			//printf(" P(%s", outcome);
			if (dimCount > 1)
			{
				printf(" | ");
				for (int j = 0; j < dimCount ; parentIdx++)
				{
					if (j > 0) printf(",");
					//DSL_node *parentNode = net.GetNode(parents[parentIdx]);
					//const DSL_idArray &parentOutcomes = *parentNode->Def()->GetOutcomeIds();
					std::cout <<  matrix[coords[j]]<< std::endl;
			}
		}*/
		//double prob = mtx[elemIdx];
	        for (int i = 0; i < matrix->GetSize(); ++i) {
             		for (int j = 0; j < matrix->GetSizeOfDimension(i); ++j) {
                	std::cout << matrix[i][j] << " ";
                	}
                std::cout << std::endl;
                }
	}
}
//map the handle of a node with the appropiate docpt
std::vector<std::pair<int,std::vector<double> > > doVec;
//map do handle to its domain size
std::vector<std::pair<int,int > > domainVec;
//char set[doDomain];
for (int i = 0; i < numDo; i ++){
        DSL_node* doNode = doX.GetNode(doArray[i].c_str());
        doHandle = doX.FindNode(doArray[i].c_str());
        const DSL_intArray &parents = doX.GetParents(doHandle);
        const DSL_idArray& doOut = *doNode->Def()->GetOutcomeIds();
        doDomain= doOut.GetSize();
        std::cout << "do node handle" << doHandle << std::endl;
        std::cout << "do node domain" << doDomain << std::endl;
        doDim*=doDomain;
	std::vector<double> docpt(doDomain, float(1.0/doDomain));
        //docpt[doVal]=1;
        //for ( int i = 0; i < doDomain; i++){
		//docpt[i]=1;
	//	docpt[i] = float(1.0/doDomain);
	//}
	domainVec.emplace_back(doHandle, doDomain);
	doVec.emplace_back(doHandle, docpt);
        std::cout << "performing intervention in true model" << std::endl;
        while(parents.GetSize() > 0){
                std::cout << " parents handle " << parents[0] << " " <<doX.GetNode(parents[0])->GetName() <<  "do handle " << doHandle <<"var name " << doNode->GetName() << std::endl;
                res = doX.RemoveArc(parents[0],doHandle);
                if (DSL_OKAY != res)
                {
                        return res;
                }
        }
}






int res1;
std::cout << "replacing cpts for all do variables " << std:: endl;
//res = doX.GetNode("V0")->Def()->SetDefinition(doXCPT);
//dovec is assigned in above code as true model so should be the same
for( int i =0; i < numDo; i ++){
	std::cout << " i " << numDo << std::endl; 
	std::cout << "replace cpt of varibale " << doVec[i].first << std::endl;
        res1 = doX.GetNode(doVec[i].first)->Def()->SetDefinition({begin(doVec[i].second), end(doVec[i].second)});
	
	doX.SetTarget(doVec[i].first, true );
}




if (DSL_OKAY != res1     )
{
	std::cout << " error " << res1 << std::endl;
	return res;
}
doX.ClearAllTargets();
std::cout << "calculating trueValue  in the mutilated true model " << std::endl;

DSL_node* sn = doX.GetNode(queryVar.c_str());
const DSL_Dmatrix& beliefs = *sn->Val()->GetMatrix();
const DSL_idArray& outcomes = *sn->Def()->GetOutcomeIds();
const DSL_nodeVal* values = sn->Val();

domain= outcomes.GetSize();
std::cout << "domain of query var" << domain << std::endl;
int solnSize = domain * doDim;
std::cout << "beliefs dim  " << beliefs.GetSize() << std::endl;
std::cout << "solution array size " << solnSize << std::endl;

//error for all queries so size of the array is going to be the obsDomain
double error[(solnSize)];
double exactSoln[(solnSize)];
//create vector of assignements
std::vector<std::vector<int>> combinations;

// Initialize the current combination
std::vector<int> current(numDo, 0);

// Generate all combinations
for (int i = 0; i < doDim; ++i) {
	combinations.push_back(current);
	j=numDo-1;
	//while (j >= 0 && ++current[j] >= domainVec[j].second) {
        //	current[j] = 0;
        //	--j;
    	//}
        // Increment the current combination
	for (int j = numDo - 1; j >= 0; --j) {
            	if (++current[j] < domainVec[j].second) {
            		break;
            	}
       		current[j] = 0;
        }
}
//DEBUG
/*
for(int i =0; i < combinations.size(); i++){
	for(int j=0; j<combinations[i].size();j++){
	       std::cout << combinations[i][j] << " ";
	}
std::cout << std::endl;
}
*/
/*for(int i =0; i < doDomain
for ( int i = 0; i < numDo; i++){
	for(int j =0; j < domainVec[i].second; j++){
		std::cout << "i,j" << i<< " " << j << std::endl;
	}
}*/
std::cout << " setting evidence " << std::endl;
int solnIndex=0;
for( int i =0; i < combinations.size(); i++){
	for ( int d =0; d < numDo; d++){	
		int evidenceNodeHandle = doVec[d].first; 
		DSL_node* evNode = doX.GetNode(evidenceNodeHandle);
		DSL_nodeVal *evVal = evNode->Val();
		const DSL_idArray& evOutcomes = *evNode->Def()->GetOutcomeIds();
		int evDomain= evOutcomes.GetSize();
		int numOut = evOutcomes.GetSize();	
			
		evVal->SetEvidence(evOutcomes[combinations[i][d]]);
	}
	
	//std::cout << "updating beliefs " << std::endl;
	doX.UpdateBeliefs();
	for (int k = 0; k < domain; k++) {
		if(print){
			printf("P(%s = %s ", sn->GetId(), (outcomes)[k]);
			std::cout << "| " << PrintVectorInfo(doX, domainVec,combinations[i]) << " ) = " ;
			std::cout << beliefs[k] << std::endl;
		}
		//	evNode->GetId(), evOutcomes[k], beliefs[i]);
		//std::cout << "print index " << d*domain*numOut+ k*numOut + i << std::endl;
		exactSoln[solnIndex] = beliefs[k];
		solnIndex++;
	}

}

/*
for ( int d =0; d < numDo; d++){	
	int evidenceNodeHandle = doVec[d].first; 
	DSL_node* evNode = doX.GetNode(evidenceNodeHandle);
	DSL_nodeVal *evVal = evNode->Val();
	const DSL_idArray& evOutcomes = *evNode->Def()->GetOutcomeIds();
	int evDomain= evOutcomes.GetSize();
	//std::vector<double> docpt(evDomain, float(1.0/evDomain));
	int numOut = evOutcomes.GetSize();
	for (int k=0; k < numOut; k++){
		
		evVal->SetEvidence(evOutcomes[k]);
		std::cout << "updating beliefs" << std::endl;
	//	doX.SetTarget(handle, true );
		doX.UpdateBeliefs();
		for (int i = 0; i < domain; i++) {
			printf("P(%s = %s |%s= %s) = %.4f\n", sn->GetId(), (outcomes)[i], evNode->GetId(), evOutcomes[k], beliefs[i]);
			//std::cout << "print index " << d*domain*numOut+ k*numOut + i << std::endl;
			exactSoln[d*domain*numOut + k*numOut+i] = beliefs[i];
		}
	}
}
*/




doX2 = DSL_network(learnedModel);



handle = doX2.FindNode(queryVar.c_str());
if (handle >= 0)
{
        printf("Handle of my query node  is: %d\n", handle);
}
else
{
        std::cout<<"There's no node with ID=myNodeId\n";
}

doVec.clear();
auto startTime = std::chrono::high_resolution_clock::now();


for (int i = 0; i < numDo; i ++){
        DSL_node* doNode = doX2.GetNode(doArray[i].c_str());
        int doHandle = doX2.FindNode(doArray[i].c_str());
        const DSL_intArray &parents = doX2.GetParents(doHandle);
        const DSL_idArray& doOut = *doNode->Def()->GetOutcomeIds();
        doDomain= doOut.GetSize();
        std::cout << "do node handle " << doHandle << std::endl;
        std::cout << "do node domain " << doDomain << std::endl;
        std::vector<double> docpt(doDomain, float(1.0/doDomain));
        doVec.emplace_back(doHandle, docpt);
        std::cout << "performing intervention in learned model" << std::endl;
        const DSL_intArray &parents2 = doX2.GetParents(doHandle);
        //std::cout << "num of Parents " << parents2.GetSize() << std::endl;
        while(parents2.GetSize() > 0){
                std::cout << " parents handle " << parents2[0] << "do handle " << doHandle << std::endl;
                res = doX2.RemoveArc(parents2[0],doHandle);
                if (DSL_OKAY != res)
                {
                        return res;
                }
        }
}


std::cout << "replacing cpts for all do variables " << std:: endl;
for( int i =0; i < numDo; i ++){
        std::cout << "replace cpt of varibale " << doVec[i].first << std::endl;
        res1 = doX2.GetNode(doVec[i].first)->Def()->SetDefinition({begin(doVec[i].second), end(doVec[i].second)});
}



if (DSL_OKAY != res1)
{
	std::cout << "error " <<res << std::endl;
	return res;
}
std::cout<< "pass" << std::endl;
res=doX2.SetTarget(handle, true );
if (DSL_OKAY != res)
{
	std::cout << "error " <<res << std::endl;
	return res;
}
double estSoln[(solnSize)];
std::cout << "performing inference " << std::endl;



DSL_node* qnode = doX2.GetNode(handle);
const DSL_Dmatrix& doX2beliefs = *qnode->Val()->GetMatrix();
const DSL_idArray& doX2outcomes = *qnode->Def()->GetOutcomeIds();


std::cout << " setting evidence " << std::endl;
solnIndex=0;
for( int i =0; i < combinations.size(); i++){
	for ( int d =0; d < numDo; d++){	
		int evidenceNodeHandle = doVec[d].first; 
		DSL_node* evNode = doX2.GetNode(evidenceNodeHandle);
		DSL_nodeVal *evVal = evNode->Val();
		const DSL_idArray& evOutcomes = *evNode->Def()->GetOutcomeIds();
		int evDomain= evOutcomes.GetSize();
		int numOut = evOutcomes.GetSize();	
			
		evVal->SetEvidence(evOutcomes[combinations[i][d]]);
	}
	
	//std::cout << "updating beliefs " << std::endl;
	doX2.SetTarget(handle, true);
	doX2.UpdateBeliefs();
	for (int k = 0; k < domain; k++) {
		if(print){
			printf("P(%s = %s ", qnode->GetId(), (doX2outcomes)[k]);
			std::cout << "| " << PrintVectorInfo(doX2, domainVec,combinations[i]) << " ) = " ;
			std::cout << beliefs[k] << std::endl;
		}
		//	evNode->GetId(), evOutcomes[k], beliefs[i]);
		//std::cout << "print index " << d*domain*numOut+ k*numOut + i << std::endl;
		estSoln[solnIndex] = doX2beliefs[k];
		solnIndex++;
	}

}

/*
for ( int d =0; d < numDo; d++){	

for ( int d =0; d < numDo; d++){	
	int evidenceNodeHandle = doVec[d].first; 
	DSL_node* evNode = doX2.GetNode(evidenceNodeHandle);
	DSL_nodeVal *evVal = evNode->Val();
	const DSL_idArray& evOutcomes = *evNode->Def()->GetOutcomeIds();
	int evDomain= evOutcomes.GetSize();
	int numOut = evOutcomes.GetSize();
	for (int k=0; k < numOut; k++){
		
		evVal->SetEvidence(evOutcomes[k]);
		std::cout << "updating beliefs" << std::endl;
		doX2.SetTarget(handle, true );
		doX2.UpdateBeliefs();
		for (int i = 0; i < domain; i++) {
			printf("P(%s = %s |%s= %s) = %.4f\n", sn->GetId(), (doX2outcomes)[i], evNode->GetId(), evOutcomes[k], doX2beliefs[i]);
			//std::cout << "print index " << d*domain*numOut+ k*numOut + i << std::endl;
			estSoln[d*domain*numOut + k*numOut+i] = doX2beliefs[i];
		}
	}
}

*/

auto endTime = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = (endTime - startTime);
std::cout << "time elapsed  for inference " << duration.count() << std::endl;

//double estExpVal=0;
double sum=0;
for (int i = 0; i < solnSize; i++)
{
        error[i] = std::fabs(exactSoln[i]- estSoln[i]);
        sum+=error[i];

}

double avgErr =0;
avgErr= sum/((solnSize));

double weighted[solnSize];
double weightedErr=0;
std::cout << "The avergae error over all values of i is  " << avgErr << std::endl;
//PrintPosteriors(doX2,handle);
for (int i=0; i < solnSize; i++){
	weighted[i]=error[i]*doMarginal[int(floor(i/weightedSize))];
	weightedErr+=weighted[i];
}

std::cout << "The weighted error is " << weightedErr << std::endl; 

outS << avgErr << std::endl;
outTinf << duration.count() << std::endl;


outTinf.close();
outS.close();


DSL_errorH().RedirectToFile(stdout);


return 0;
}


static void PrintMatrix(DSL_network net,  const DSL_Dmatrix &mtx, const DSL_idArray &outcomes, const DSL_intArray &parents){
	int dimCount = mtx.GetNumberOfDimensions();
	DSL_intArray coords(dimCount);
	coords.FillWith(0);
	
	for (int elemIdx = 0; elemIdx < mtx.GetSize(); elemIdx++)
	{
		const char *outcome = outcomes[coords[dimCount - 1]];
		printf(" P(%s", outcome);
		if (dimCount > 1)
		{
			printf(" | ");
			for (int parentIdx = 0; parentIdx < dimCount - 1; parentIdx++)
			{
				if (parentIdx > 0) printf(",");
				DSL_node *parentNode = net.GetNode(parents[parentIdx]);
				const DSL_idArray &parentOutcomes = *parentNode->Def()->GetOutcomeIds();
				printf("%s=%s",parentNode->GetId(), parentOutcomes[coords[parentIdx]]);
			}
		}
		double prob = mtx[elemIdx];
		printf(")=%g\n", prob);
		mtx.NextCoordinates(coords);
	}
}


// PrintNodeInfo displays node attributes:
// name, outcome ids, parent ids, children ids, CPT probabilities
static void PrintNodeInfo(DSL_network &net, int nodeHandle)
{
	DSL_node *node = net.GetNode(nodeHandle);
	printf("Node: %s\n", node->GetName());
	printf(" Outcomes:");
	const DSL_idArray &outcomes = *node->Def()->GetOutcomeIds();
	for (const char* oid : outcomes)
	{
		printf(" %s", oid);
	}
	printf("\n");
	const DSL_intArray &parents = net.GetParents(nodeHandle);
	if (!parents.IsEmpty())
	{
		printf(" Parents:");
		for (int p: parents)
		{	
			printf(" %s", net.GetNode(p)->GetId());
		}
		printf("\n");
	}
	const DSL_intArray &children = net.GetChildren(nodeHandle);
	if (!children.IsEmpty())
	{
		printf(" Children:");
		for (int c: children)
		{
			printf(" %s", net.GetNode(c)->GetId());
		}
		printf("\n");
	}
	const DSL_nodeDef *def = node->Def();
	int defType = def->GetType();
	printf(" Definition type: %s\n", def->GetTypeName());
	if (DSL_CPT == defType || DSL_TRUTHTABLE == defType)
	{
		const DSL_Dmatrix &cpt = *def->GetMatrix();
		PrintMatrix(net, cpt, outcomes, parents);
	}
}

static int CreateCptNode(DSL_network &net, const char *id, const char *name, 
    std::initializer_list<const char *> outcomes, int xPos, int yPos)
{

    int handle = net.AddNode(DSL_CPT, id);

    DSL_node *node = net.GetNode(handle);

    node->SetName(name);

    node->Def()->SetNumberOfOutcomes(outcomes);

    DSL_rectangle &position = node->Info().Screen().position;

    position.center_X = xPos;

    position.center_Y = yPos;

    position.width = 85;

    position.height = 55;

    return handle;

}


static void PrintPosteriors(DSL_network &net, int handle)
{
	std::cout << "printing posteriors "<< std::endl;
	DSL_node *node = net.GetNode(handle);
	const char* nodeId = node->GetId();
	const DSL_nodeVal* val = node->Val();
	if (val->IsEvidence())
	{
		printf("%s has evidence set (%s)\n", nodeId, val->GetEvidenceId());
	}
	else
	{
		const DSL_idArray& outcomeIds = *node->Def()->GetOutcomeIds();
		const DSL_Dmatrix& posteriors = *val->GetMatrix();
		std::cout << "posteriors size " << posteriors.GetSize() << std::endl;
		for (int i = 0; i < posteriors.GetSize(); i++)
		{
			printf("P(%s=%s)=%g\n", nodeId, outcomeIds[i], posteriors[i]);
		}		
	}
}



std::string PrintVectorInfo(DSL_network &net, std::vector<std::pair<int,int > > domainVec, std::vector<int> currentCombo ){
	std::string outString="";
	for( int n =0; n < domainVec.size(); n++){
		DSL_node *node = net.GetNode(domainVec[n].first);
		const char* nodeId = node->GetId();
		const DSL_nodeVal* val = node->Val();
		const DSL_idArray& outcomes= *node->Def()->GetOutcomeIds();
		outString+= std::string(nodeId) + " =  ";
	       	outString+= std::string(outcomes[currentCombo[n]]) + " ";

	}
	return outString;
}

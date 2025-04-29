//g++ -DNDEBUG -O3 learn_main.cpp -o learn.out -I./smile -L./smile -lsmile
//usage ./learn.out model_file.xdsl em_model.xdsl datafile.csv  numPts
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
#include <filesystem> // Add this include at the top of the file

//using namespace std;

static int CreateCptNode(DSL_network &net, const char *id, const char *name, 
    std::initializer_list<const char *> outcomes, int xPos, int yPos);

static void PrintPosteriors(DSL_network &net, int handle);

int main(int argc, char *argv[]){

DSL_network net;
DSL_network emModel;
DSL_network doX;
DSL_network doX2;
DSL_dataset ds;
std::vector<DSL_datasetMatch> matching;
std::string errMsg;
std::ofstream outTinf;
std::ofstream outTlearn;
std::ofstream outS;
std::ofstream outLL;
std::ofstream outSeed;
double loglik;
int seed =0;
std::string outPath;
std::string seedPath="";
std::string modelFile = "";
std::string emFile = "";
std::string dataFile = "";
std::string learnedFile="";
std::string basename;
std::size_t dot=0;//unisigned integer
std::size_t slash=0;
std::string queryVar;
std::string doVar;
int numPts=0;
int hypDomain= 0;
int domain;
int doDomain;
double totalError =0;
double avgError =0;
int doVal=0;
int numDo;
if(argc <5)
{
	std::cout << "not enough command line arguments passed" << std::endl;
}
else{
	modelFile = argv[1];
	emFile =  argv[2];
	dataFile = argv[3];
	numPts=std::stoi(argv[4]);
}
//doVar= argv[5];
int argvIn=5;
int j=0;
//seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

dot = modelFile.find('.');
slash = modelFile.rfind('/');
basename=modelFile.substr(slash+1,dot-(slash+1));
outPath = "learned_models/" + basename + "/"+ std::to_string(numPts)+"/";
dot = emFile.find('.');
slash = emFile.rfind('/');

// Ensure the output directory exists
std::filesystem::create_directories("learned_models/" + basename + "/" + std::to_string(numPts));

learnedFile = "learned_models/" + basename + "/" + std::to_string(numPts) + "/" +
              emFile.substr(slash + 1, dot - (slash + 1)) + ".xdsl";

basename = emFile.substr(slash + 1, dot - (slash + 1) - 2);
outPath += basename + "/";
std::filesystem::create_directories(outPath);  // Ensure the additional subdirectory is created
std::cout <<"learned file " << learnedFile << std::endl;
std::cout << "outpath " << outPath << std::endl;
outTlearn.open(outPath+"timesLearn.csv", std::ios_base::app);
outLL.open(outPath+"LL.csv", std::ios_base::app);
//outSeed.open(outPath+"seed.csv", std::ios_base::app);


int res = net.ReadFile(modelFile.c_str());
std::cout << "the model file is : " << modelFile.c_str() << std::endl;
if (DSL_OKAY != res)
{
	return res;
}


res = ds.ReadFile(dataFile.c_str());
if (DSL_OKAY != res)
{
	return res;
}

doX = DSL_network(net);




res = emModel.ReadFile(emFile.c_str()); if (DSL_OKAY != res)
{
	std::cout << "error reading em file "<< res <<std::endl;
	return res;
}

DSL_errorH().RedirectToFile(stdout);
auto startTime = std::chrono::high_resolution_clock::now();


res = ds.MatchNetwork(emModel, matching, errMsg);
if (DSL_OKAY == res)
{
	DSL_em em;
	em.SetRelevance(true);
//optional 
//      em.SetSeed(seed)
	em.SetRandomizeParameters(true);
//	seed=em.GetSeed();
	res = em.Learn(ds, emModel, matching, &loglik);
}

auto endTime = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = (endTime - startTime);
std::cout << "time elapsed for learning " << duration.count() << std::endl;
outTlearn << duration.count() << std::endl;
emModel.WriteFile(learnedFile.c_str());

startTime = std::chrono::high_resolution_clock::now();



outLL << loglik << std::endl;
outSeed << seed << std::endl;

outSeed.close();
outTlearn.close();
outLL.close();

//posteriors = doX2.get_node_value(queryVar)
//print(posteriors[0])

DSL_errorH().RedirectToFile(stdout);


return 0;
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
	DSL_node *node = net.GetNode(handle);
	const char* nodeId = node->GetId();
	const DSL_nodeVal* val = node->Val();
	if (val->IsEvidence())
	{
		printf("%s has evidence set (%s)\n",
		nodeId, val->GetEvidenceId());
	}
	else
	{
		const DSL_idArray& outcomeIds = *node->Def()->GetOutcomeIds();
		const DSL_Dmatrix& posteriors = *val->GetMatrix();
		for (int i = 0; i < posteriors.GetSize(); i++)
		{
			printf("P(%s=%s)=%g\n", nodeId, outcomeIds[i], posteriors[i]);
		}		
	}
}


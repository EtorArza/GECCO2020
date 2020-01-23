//#define SAME_SIZE_EXPERIMENT

/*
  Copyright 2001 The University of Texas at Austin

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "INIReader.h"
#include <unistd.h>
#include "experiment.h"
#include "neat.h"
#include "rng.h"
#include "util.h"
#include <omp.h>
#include "loadnetworkfromfile.h"
#include "Population.h"
#include "PBP.h"
#include "QAP.h"
#include "LOP.h"
#include "PFSP.h"
#include "TSP.h"
#include "permuevaluator.h"
#include "Tools.h"
#include "FitnessFunction_permus.h"


#define EXTERN
#include "Parameters.h"
#include <iomanip>      // std::setprecision


using namespace NEAT;
using namespace std;



void usage()
{

    cerr << "usage: \n ./neat path_of_config_file" << endl;
    cerr << "for example, \n ./neat \"config_files/test.ini\"" << endl;
    cerr << endl;
    exit(1);
}


PBP *GetProblemInfo(std::string problemType, std::string filename)
{
    PBP *problem;
    if (problemType == "pfsp")
    {
        problem = new PFSP();
    }
    else if (problemType == "tsp")
    {
        problem = new TSP();
    }
    else if (problemType == "qap")
    {
        problem = new QAP();
    }
    else if (problemType == "lop")
    {
        problem = new LOP();
    }
    // else if (problemType == "api")
    //     problem = new API();
    else
    {
         cout << "Wrong problem type was specified." << endl;
         exit(1);
     }

    //Read the instance.
    problem->Read_with_mutex(filename);
    #ifdef SAME_SIZE_EXPERIMENT
        if (problem->GetProblemSize() == 30)
        {
            MAX_TIME_PSO = 0.10;
        }
        else if(problem->GetProblemSize() == 60)
        {
            MAX_TIME_PSO = 0.3;
        }
        else
        {
            cout << "ERROR, this experiment expects instances of size 60 and 30.";
            exit(1);
        }

    #undef SAME_SIZE_EXPERIMENT
    #endif
    return problem;
}



int main(int argc, char *argv[])
{   

    // int n = 10;

    // double* array_1 = new double[n];
    // double* order_res = new double[n];

    // for (int i = 0; i < n; i++)
    // {
    //     array_1[i] = (double) i+10;
    // }

    // array_1[0] = 11.0;
    // array_1[3] = 10.0;
    // array_1[7] = 18.0;
    // array_1[6] = 18.0;

    // PrintArray(array_1, n);

    // compute_order_from_double_to_double(array_1, n, order_res, false, true);

    // PrintArray(array_1, n);

    // PrintArray(order_res, n);

    // exit(1);

    #ifndef NDEBUG
    cout << "WARNING: Debug mode. Assertions enabled." << endl;
    #endif

    if (argc < 2)
    {
        cout << "Error, no configuration file provided.\n";
        exit(1);
    }else if (argc > 2)
    {
        cout << "Error, too many arguments provided. Provide only path to configuration file.";
        exit(1);

    }

    INIReader reader(argv[1]);

    if (reader.ParseError() != 0) {
        std::cout << "Can't load " << argv[1] << "\n";
        return 1;
    }

    string MODE = reader.Get("Global", "mode", "UNKNOWN");



    if (MODE ==  "test")
    {

        //const char * prob_name = "permu";
        //Experiment *exp = Experiment::get(prob_name);

        PROBLEM_TYPE = reader.Get("Controller", "PROBLEM_TYPE", "UNKOWN");
        INSTANCE_PATH = reader.Get("Controller", "PROBLEM_PATH", "UNKOWN");
        MAX_TIME_PSO = reader.GetReal("Controller", "MAX_TIME_PSO", -1.0);
        POPSIZE = reader.GetInteger("Controller", "POPSIZE", -1);
        TABU_LENGTH = reader.GetInteger("Controller", "TABU_LENGTH", -1);
        CONTROLLER_PATH = reader.Get("TestSettings", "CONTROLLER_PATH", "UNKNOWN");
        N_REPS = reader.GetInteger("TestSettings", "N_REPS", -1);
        N_EVALS = reader.GetInteger("TestSettings", "N_EVALS", -1);
        N_OF_THREADS = reader.GetInteger("TestSettings", "THREADS", 1);
        N_OF_THREADS = min(N_OF_THREADS, N_EVALS);


        if (CONTROLLER_PATH == "UNKNOWN")
        {
            cout << "error, controller path not specified in test." << endl;
        }

        if (N_REPS < 0)
        {
             cout << "error, N_REPS not provided in test mode." << endl;
        }
        
        
        CpuNetwork net = load_network(CONTROLLER_PATH);


        cout << std::setprecision(15);
        RandomNumberGenerator* rng;
        rng = new RandomNumberGenerator();
        rng->seed();
        int initial_seed = rng->random_integer_uniform(40000000, 50000000);
        delete rng;
        cout << "[[";
        for (int j = 0; j < N_REPS; j++)
        {

            double res = FitnessFunction_permu(&net, N_EVALS, initial_seed);
            initial_seed += N_EVALS;
            cout << res;
            if (j < N_REPS-1)
            {
                cout << ",";
            }
            
        }
        cout << "]," << std::flush;



        

        cout << std::setprecision(15);
        cout << std::flush;
        cout << INSTANCE_PATH   << "," 
             << CONTROLLER_PATH << "," 
             << PROBLEM_TYPE    << "," 
             << N_EVALS
             << "]"
             << endl;


        // cout << res << std::endl;;
        cout << std::flush;

        return 0;
    }
    else
    {
        cout << "invalid mode provided. Please, use the configuration file to specify either test or train." << endl;
        exit(1);
    }
}

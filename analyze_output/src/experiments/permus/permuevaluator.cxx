#include "std.hxx"

#include "Parameters.h"
#include "permuevaluator.h"
#include "FitnessFunction_permus.h"
#include "map.h"
#include "network.h"
#include "networkexecutor.h"
#include "resource.h"
#include <assert.h>
#include "PBP.h"
#include "Population.h"
#include "Tools.h"
#include <cfloat>

using namespace std;

//#define COUNTER
//#define PRINT
//#define RANDOM_SEARCH
#define WRITE_OUTPUTS

#ifdef WRITE_OUTPUTS
#define TIME_CLASSES 1
#define QUALITY_CLASSES 1
#endif


PBP *GetProblemInfo(std::string problemType, std::string filename);



void transform_output_to_meaningful_representation(double* neat_out, double* meaningful_representation)
{



    if (neat_out[0] > CUTOFF_0)
    {
        meaningful_representation[0] = 1.0;
    }
    else if(neat_out[0] < -CUTOFF_0)
    {
        meaningful_representation[0] = -1.0;
    }
    else
    {
        for (int i = 0; i < NEAT::__output_N; i++)
        {
            meaningful_representation[i] = 0.0;
        }
        return;
    }

    #define INDEX_OF_FIRST_OPERATOR_OUT 1

    int index_largest = argmax(neat_out+INDEX_OF_FIRST_OPERATOR_OUT, NEAT::N_OPERATORS);

    meaningful_representation[1] = 0.0;
    meaningful_representation[2] = 0.0;
    meaningful_representation[3] = 0.0;
    meaningful_representation[index_largest+INDEX_OF_FIRST_OPERATOR_OUT] = 1.0;

    meaningful_representation[NEAT::accept_or_reject_worse] = neat_out[NEAT::accept_or_reject_worse] / 2.0;
    meaningful_representation[NEAT::accept_or_reject_worse] += 0.5;


    if (neat_out[NEAT::TABU] > CUTOFF_0)
    {
        meaningful_representation[NEAT::TABU] = 1.0;
    }
    else if(neat_out[NEAT::TABU] < -CUTOFF_0)
    {
        meaningful_representation[NEAT::TABU] = 0.0;
    }
    else
    {
        meaningful_representation[NEAT::TABU] = 0.0;
    }


    #define INDEX_OF_FIRST_COEFFICIENT_OUT 6

    index_largest = abs_argmax(neat_out + INDEX_OF_FIRST_COEFFICIENT_OUT, NEAT::N_COEF);

    for (int i = 0; i < NEAT::N_COEF; i++)
    {
        meaningful_representation[i + INDEX_OF_FIRST_COEFFICIENT_OUT] = 0.0;
    }

    meaningful_representation[index_largest + INDEX_OF_FIRST_COEFFICIENT_OUT] = 1.0;

}

double FitnessFunction_permu(NEAT::CpuNetwork *net_original, int n_evals, int seed)
{

    double *v_of_fitness;
    PBP *problem;
    CPopulation *pop;

    NEAT::CpuNetwork tmp_net = *net_original;
    NEAT::CpuNetwork *net = &tmp_net;

    problem = GetProblemInfo(PROBLEM_TYPE, INSTANCE_PATH); //Read the problem instance to optimize.
    pop = new CPopulation(problem);
    problem->load_rng(pop->rng);
    pop->rng->seed(seed);

    v_of_fitness = new double[n_evals];

    for (int i = 0; i < POPSIZE; i++)
    {
        pop->m_individuals[i]->activation = std::vector<double>(net->activations);
    }

    double ***average_output_matrix = new double **[QUALITY_CLASSES]; // [Position_in_pop][Position_in_time][position_in_output]

    for (int i = 0; i < QUALITY_CLASSES; i++)
    {
        average_output_matrix[i] = new double *[TIME_CLASSES];
        for (int j = 0; j < TIME_CLASSES; j++)
        {
            average_output_matrix[i][j] = new double[NEAT::__output_N];
            for (int k = 0; k < NEAT::__output_N; k++)
            {
                average_output_matrix[i][j][k] = 0.0;
            }
        }
    }

    std::string output_filename = "output_values.txt";
    append_line_to_file(output_filename, from_path_to_filename(CONTROLLER_PATH) + "|" + from_path_to_filename(INSTANCE_PATH) + "\n");

    double output_matrix[QUALITY_CLASSES][TIME_CLASSES][NEAT::__output_N] = {0}; // [Position_in_pop][Position_in_time][position_in_output]


    for (int n_of_repetitions_completed = 0; n_of_repetitions_completed < n_evals; n_of_repetitions_completed++)
    {

        int counter = 0;

        int start_phase = 0;   // number of iterations done in previous phase.
        int last_time_pos = 0; // current phase.
        int time_pos = 0;

        for (int i = 0; i < QUALITY_CLASSES; i++)
        {
            for (int j = 0; j < TIME_CLASSES; j++)
            {
                for (int k = 0; k < NEAT::__output_N; k++)
                {
                    output_matrix[i][j][k] = 0.0;
                }
            }
        }

        pop->rng->seed(seed + n_of_repetitions_completed);
        pop->Reset();
        //std::cout << "|" << n_of_repetitions_completed << "|" << std::endl;
        for (int i = 0; i < POPSIZE; i++)
        {
            std::swap(net->activations, pop->m_individuals[i]->activation);
            net->clear_noninput();
            std::swap(net->activations, pop->m_individuals[i]->activation);
        }



        while (!pop->terminated)
        {
            counter++;
            for (int i = 0; i < POPSIZE; i++)
            {

                std::swap(net->activations, pop->m_individuals[i]->activation);
                for (int sns_idx = 0; sns_idx < NEAT::__sensor_N; sns_idx++)
                {
                    net->load_sensor(sns_idx, pop->get_neat_input_individual_i(i)[sns_idx]);
                }
                net->activate();
                pop->apply_neat_output_to_individual_i(net->get_outputs(), i);

                double *meaningful_outs = new double[NEAT::__output_N];

                for (int idx = 0; idx < NEAT::__output_N; idx++)
                {
                    meaningful_outs[idx] = 0.0;
                }
                

                transform_output_to_meaningful_representation(net->get_outputs(), meaningful_outs);
                int pop_pos = QUALITY_CLASSES * i / POPSIZE;
                sum_arrays(&output_matrix[pop_pos][time_pos][0], &output_matrix[pop_pos][time_pos][0], meaningful_outs, NEAT::__output_N);
                delete[] meaningful_outs;

                std::swap(net->activations, pop->m_individuals[i]->activation);
            }
            pop->end_iteration();

            time_pos = (int)((double)TIME_CLASSES * pop->timer->toc() / MAX_TIME_PSO);
            time_pos = min(time_pos, TIME_CLASSES - 1);
            if (last_time_pos != time_pos || pop->terminated)
            {
                int n_permus_in_each_quantile = POPSIZE / QUALITY_CLASSES;
                if (POPSIZE % QUALITY_CLASSES != 0)
                {
                    cout << "ERROR, Quality classes needs to divide POPPSIZE evenly. " << endl;
                    exit(1);
                }
                int n_iters_in_this_positon = counter - start_phase;
                start_phase = counter;

                for (int pop_pos = 0; pop_pos < QUALITY_CLASSES; pop_pos++)
                {
                    multiply_array_with_value(&output_matrix[pop_pos][last_time_pos][0], 1.0 / (double)(n_iters_in_this_positon * n_permus_in_each_quantile), NEAT::__output_N);
                }
                last_time_pos++;
            }
            //pop->Print();
        }
        if (!isPermutation(pop->genome_best, pop->n))
        {
            cout << "final result is not permutation" << endl;
            cout << "final permu: ";
            PrintArray(pop->genome_best, pop->n);
            exit(1);
        }
        v_of_fitness[n_of_repetitions_completed] = problem->Evaluate(pop->genome_best);
        net->clear_noninput();


        for (int i = 0; i < QUALITY_CLASSES; i++)
        {
            for (int j = 0; j < TIME_CLASSES; j++)
            {
                sum_arrays(average_output_matrix[i][j], average_output_matrix[i][j], &output_matrix[i][j][0], NEAT::__output_N);
            }
        }
    }

    double res = Average(v_of_fitness, n_evals);



    for (int i = 0; i < QUALITY_CLASSES; i++)
    {
        for (int j = 0; j < TIME_CLASSES; j++)
        {
            multiply_array_with_value(&average_output_matrix[i][j][0], (1.0 / (double)n_evals), NEAT::__output_N);
            std::string str_array = to_string(i) + "|" + to_string(j) + "|" + array_to_string(&average_output_matrix[i][j][0], NEAT::__output_N) + "\n";
            append_line_to_file(output_filename, str_array);
        }
    }

    for (int i = 0; i < QUALITY_CLASSES; i++)
    {
        for (int j = 0; j < TIME_CLASSES; j++)
        {
            delete[] average_output_matrix[i][j];
        }
        delete[] average_output_matrix[i];
    }
    delete[] average_output_matrix;

    delete[] v_of_fitness;
    delete pop;
    delete problem;
    pop = NULL;
    v_of_fitness = NULL;
    problem = NULL;
    net = NULL;
    return res;
}

namespace NEAT
{
struct Config
{
};
struct Evaluator
{
    typedef NEAT::Config Config;
    const Config *config;


    __net_eval_decl Evaluator(const Config *config_) : config(config_){};


    // fitness function in sequential order
    __net_eval_decl double FitnessFunction(CpuNetwork *net, int n_evals, int initial_seed)
    {
        int seed_seq = initial_seed;
        double res = FitnessFunction_permu(net, n_evals, seed_seq);
        seed_seq += n_evals;
        return res;
    }

    // parallelize over the same network
    __net_eval_decl void FitnessFunction_parallel(CpuNetwork *net, int n_evals, double *res, int initial_seed)
    {
        int seed_parallel = initial_seed;

        #pragma omp parallel for num_threads(N_OF_THREADS)
        for (int i = 0; i < n_evals; i++)
        {
            res[i] = FitnessFunction_permu(net, N_EVALS, seed_parallel + i);
        }
        seed_parallel += n_evals;
    }
};

class PermuEvaluator : public NetworkEvaluator
{
    NetworkExecutor<Evaluator> *executor;

public:
    PermuEvaluator()
    {
        executor = NetworkExecutor<Evaluator>::create();
        //Evaluator::Config *config;

        // size_t configlen;
        // create_config(config, configlen);
        // executor->configure(config, configlen);
        // free(config);
    }

    ~PermuEvaluator()
    {
        delete executor;
    }

    virtual void execute(class Network **nets_,
                         class OrganismEvaluation *results,
                         size_t nnets)
    {
        executor->execute(nets_, results, nnets);
    }
};

class NetworkEvaluator *create_permu_evaluator()
{
    return new PermuEvaluator();
}

} // namespace NEAT

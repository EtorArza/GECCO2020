# Instructions to replicate the experimentation of the GECCO2020 paper(using GCC in ubuntu 18.04):


## For an up to date version of the methodology introduced in this paper, check out https://github.com/EtorArza/NeatPerturbHH
The licence is also available in the other repo. Please use this repo only to reproduce the results, if you want to use my work please use the up to date code on the other repo.


## Experiment on the behaviour of the controllers

1) change directory to GECCO20/analyze_output/
    
    `cd analyze_output/`

2) execute "bash scripts/release_compile.sh"
    
    `bash scripts/release_compile.sh`  


### Step 3.1 and 3.2 are optional, but both or none of them must be done

3.1) delete output_values.txt

3.2) execute "bash scripts/measure_behaviour_of_all_pairs.sh"

4) execute `python scripts/obtain_figures_with_output_values.py`

## Experiment on the quality of the controllers

1) This experiment was conducted on a multi node server with 32 core nodes with slurm. Thus, both the training and the testing are executed in the server. The scripts inside GECCO/train_test_controllers/scripts are used.

Once the file GECCO2020/train_test_controllers/result_controllers.txt has been obtained, 

2) `cd GECCO2020/train_test_controllers/`

3) execute `python train_test_controllers/scripts/plot_square_ranking_matrix.py`


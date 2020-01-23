for controller in experiments/transfer_qap_with_cut_instances/top_controllers/*; do
    [ -e "$controller" ] || continue
    for instance in experiments/transfer_qap_with_cut_instances/instances/cut_instances/*; do
        [ -e "$instance" ] || continue

controller_name=$(basename ${controller})
instance_name=$(basename ${instance})

echo "${controller_name} - ${instance_name}"

cat > "tmp${controller_name}${instance_name}.ini" <<EOF


[Global]
mode = test


[TestSettings]
THREADS = 1
N_EVALS = 40
N_REPS = 1
CONTROLLER_PATH = ${controller}


[Controller]
MAX_TIME_PSO = 0.15
POPSIZE = 20
TABU_LENGTH = 40
PROBLEM_TYPE = qap
PROBLEM_PATH = ${instance}

EOF


./neat "tmp${controller_name}${instance_name}.ini"

rm "tmp${controller_name}${instance_name}.ini"


    done
done






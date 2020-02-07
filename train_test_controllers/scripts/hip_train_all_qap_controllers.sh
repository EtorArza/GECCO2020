    for instance in instances/*; do
        [ -e "$instance" ] || continue
        sbatch scripts/exec_hipatia_train.sh "qap" "$instance"
    done




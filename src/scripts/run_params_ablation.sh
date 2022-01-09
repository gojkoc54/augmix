# Excluding depth=4 and width=3 since it is covered by the default values
widths=( 1 2 4 5 )
depths=( 1 2 3 )
epochs=30
num_workers=0

echo 'Ablation study execution log' > params_ablation_log.txt

# Executing the standard case without augmix
printf "\n\nExecuting without augmix \n\n"
echo "NO augmix" >> params_ablation_log.txt
python3 main.py --should-augmix 0 --mixture-width 0 --mixture-depth 0 \
    --epochs ${epochs} --num-workers ${num_workers}
    
# Executing AugMix with default parameters 
printf "\n\nExecuting augmix with default parameters (WIDTH = 3, DEPTH = 4)\n\n"
echo "width = 3; depth = 4" >> params_ablation_log.txt
python3 main.py --epochs ${epochs} --num-workers ${num_workers}
echo "done" >> params_ablation_log.txt

# Varying the width while keeping the depth default (4)
for width in "${widths[@]}"
do
    printf "\n\nExecuting for parameters: WIDTH = ${width} DEPTH = 4 \n\n"
    echo "width = ${width}; depth = 4" >> params_ablation_log.txt
    python3 main.py --mixture-width ${width} \
        --epochs ${epochs} --num-workers ${num_workers}
    echo "done" >> params_ablation_log.txt
done

# Varying the depth while keeping the width default (3)
for depth in "${depths[@]}"
do
    printf "\n\nExecuting for parameters: WIDTH = 3 DEPTH = ${depth} \n\n"
    echo "width = 3; depth = ${depth}" >> params_ablation_log.txt
    python3 main.py --mixture-depth ${depth} \
        --epochs ${epochs} --num-workers ${num_workers}
    echo "done" >> params_ablation_log.txt
done
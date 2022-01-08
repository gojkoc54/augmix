widths=( 1 2 3 4 5 )
depths=( 1 2 3 ) # 4 ) 
# Excluding depth=4 since it is covered by width=3 and depth=default (4) 
epochs=30

echo 'Ablation study execution log' > params_ablation_log.txt

# Executing the default case without augmix
printf "\n\nExecuting without augmix \n\n"
echo "NO augmix" >> params_ablation_log.txt
python3 main.py --should-augmix 0 --epochs ${epochs}
echo "done" >> params_ablation_log.txt

# # Varying the width while keeping the depth default (4)
# for width in "${widths[@]}"
# do
#     printf "\n\nExecuting for parameters: WIDTH = ${width} DEPTH = 4 \n\n"
#     echo "width = ${width}; depth = 4" >> params_ablation_log.txt
#     python3 main.py --mixture-width ${width} --epochs ${epochs}
#     echo "done" >> params_ablation_log.txt
# done

# # Varying the depth while keeping the width default (3)
# for depth in "${depths[@]}"
# do
#     printf "\n\nExecuting for parameters: WIDTH = 3 DEPTH = ${depth} \n\n"
#     echo "width = 3; depth = ${depth}" >> params_ablation_log.txt
#     python3 main.py --mixture-depth ${depth} --epochs ${epochs}
#     echo "done" >> params_ablation_log.txt
# done

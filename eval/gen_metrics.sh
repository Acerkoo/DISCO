plm_list=(
    gpt-j-6b
    llama-2-7b
    llama-2-13b
)
dataset_list=(
    zsre 
    counterfact
)

method_list=(
    FT
    IKE
    DISCO_k1_w1
    ROME
    MEMIT
)


for plm in ${plm_list[@]}; do
for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        echo "=====" $plm $dataset ${method} "====="
        sh eval.sh $plm $dataset ${method}
        sh port_eval.sh $plm $dataset ${method}
    done
done
done

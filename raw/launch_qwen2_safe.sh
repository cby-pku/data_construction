model_list=(
    /cpfs/user/boyuan/models/qwen2/Qwen2.5-7B-Instruct
    /cpfs/user/boyuan/models/qwen2/Qwen2-72B-Instruct
    # you can replace it with other models
    # or add more models below
)

# set gpu to what you want
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | cut -d" " -f1 | cut -d":" -f1 | tr '\n' ',' | sed 's/,$//')

INPUTFILE="/cpfs/user/boyuan/workspace/aligner-reproduce/dataset/aligner_zh_20k/zh_prompts.json"

NUM_RESPONSES=1
TASK_TEMPLATE="blank"
MODEL_TEMPLATE="qwen2"

for model in "${model_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    OUTPUT_DIR="/cpfs/user/boyuan/workspace/aligner-reproduce/dataset/aligner_zh_20k" # model name to change
    OUTPUT_NAME=${model_name} # model name to change

    bash generation.sh \
        --model_name_or_path ${model} \
        --output_dir ${OUTPUT_DIR} \
        --output_name ${OUTPUT_NAME} \
        --input_path ${INPUTFILE} \
        --num_responses ${NUM_RESPONSES} \
        --task_template ${TASK_TEMPLATE} \
        --model_template ${MODEL_TEMPLATE}
done 

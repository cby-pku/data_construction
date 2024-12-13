model_list=(
    # /cpfs/user/boyuan/models/qwen2/Qwen2.5-7B-Instruct
    # /cpfs/user/boyuan/models/qwen2/Qwen2-72B-Instruct
    #  /cpfs/user/boyuan/models/baichuan/Baichuan2-13B-Chat
    # /cpfs/user/boyuan/models/llama3/Llama-3.1-8B-Instruct
    # /cpfs/user/boyuan/models/llama3.1-ch/Llama3.1-8B-Chinese-Chat
    # /cpfs/user/boyuan/models/glm/glm-4-9b-chat
    /cpfs/user/boyuan/models/yi/Yi-1.5-34B-Chat
    # you can replace it with other models
    # or add more models below
)

# 动态设置所有可用的GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | cut -d" " -f1 | cut -d":" -f1 | tr '\n' ',' | sed 's/,$//')
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# INPUTFILE="/cpfs/user/boyuan/workspace/aligner-reproduce/dataset/aligner_zh_20k/zh_prompts.json"
INPUTFILE="/cpfs/user/boyuan/workspace/aligner-reproduce/generate_scripts/test/daily_dialogue_test.json"

NUM_RESPONSES=1
TASK_TEMPLATE="blank"
MODEL_TEMPLATE="yi"
export TRUST_REMOTE_CODE=1

for model in "${model_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    # OUTPUT_DIR="/cpfs/user/boyuan/workspace/aligner-reproduce/dataset/aligner_zh_20k" # model name to change
    OUTPUT_DIR="/cpfs/user/boyuan/workspace/aligner-reproduce/generate_scripts/test" # model name to change
    OUTPUT_NAME=${model_name} # model name to change

    yes | bash generation.sh \
        --model_name_or_path ${model} \
        --output_dir ${OUTPUT_DIR} \
        --output_name ${OUTPUT_NAME} \
        --input_path ${INPUTFILE} \
        --num_responses ${NUM_RESPONSES} \
        --task_template ${TASK_TEMPLATE} \
        --model_template ${MODEL_TEMPLATE}
done 

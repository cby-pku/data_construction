model_list=(
    # /cpfs/user/boyuan/models/qwen2/Qwen2.5-7B-Instruct
    # /cpfs/user/boyuan/models/qwen2/Qwen2-72B-Instruct
    # /cpfs/user/boyuan/models/baichuan/Baichuan2-13B-Chat
    # you can replace it with other models
    # or add more models below
    # /cpfs/user/boyuan/models/glm/glm-4-9b-chat
    /aifs4su/yaodong/spring_r1_model/Yi-1.5-34B-Chat
)

# 动态设置所有可用的GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"

INPUTFILE="/aifs4su/yaodong/spring_r1/boyuan_eval/align-anything/aligner_workspace/data_construction/questions/safe_question_set.json"
# INPUTFILE="/cpfs/user/boyuan/workspace/aligner-reproduce/generate_scripts/test/daily_dialogue_test.json"

NUM_RESPONSES=1
TASK_TEMPLATE="blank"
MODEL_TEMPLATE="yi"
export TRUST_REMOTE_CODE=1

for model in "${model_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    OUTPUT_DIR="/aifs4su/yaodong/spring_r1/boyuan_eval/align-anything/aligner_workspace/test_dataset/qa" # model name to change
    # OUTPUT_DIR="/cpfs/user/boyuan/workspace/aligner-reproduce/generate_scripts/test" # model name to change
    OUTPUT_NAME=${model_name} # model name to change

    # 使用 yes 命令自动提供 "y" 输入
    yes | bash generation.sh \
        --model_name_or_path ${model} \
        --output_dir ${OUTPUT_DIR} \
        --output_name ${OUTPUT_NAME} \
        --input_path ${INPUTFILE} \
        --num_responses ${NUM_RESPONSES} \
        --task_template ${TASK_TEMPLATE} \
        --model_template ${MODEL_TEMPLATE}
done 

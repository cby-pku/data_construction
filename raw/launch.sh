file_list=(
    /cpfs/user/boyuan/models/Alpaca3-8B
    # you can replace it with other models
    # or add more models below
)

# set gpu to what you want
export CUDA_VISIBLE_DEVICES="0,1,2,3"

INPUTFILE="/cpfs/user/boyuan/workspace/red-o1/generate_scripts/daily_dialogue_test.json"

NUM_RESPONSES=1
TASK_TEMPLATE="dialogue"
MODEL_TEMPLATE="dialogue"

for model in "${file_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    OUTPUT_DIR="/cpfs/user/boyuan/workspace/red-o1/evaluation/test_daily_dialogue/" # model name to change
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

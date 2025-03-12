model_list=(
    /aifs4su/yaodong/models/Qwen2.5/Qwen2.5-7B-Instruct
    /aifs4su/yaodong/models/QVQ-72B-Preview
    # you can replace it with other models
    # or add more models below
)

# set gpu to what you want
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

INPUTFILE="/aifs4su/yaodong/spring_r1/boyuan_eval/align-anything/aligner_workspace/data_construction/questions/safe_question_set.json"

NUM_RESPONSES=1
TASK_TEMPLATE="blank"
MODEL_TEMPLATE="qwen2"

for model in "${model_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    OUTPUT_DIR="/aifs4su/yaodong/spring_r1/boyuan_eval/align-anything/aligner_workspace/test_dataset/qa" # model name to change
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

import json

def convert_keys(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换键名
    converted_data = []
    for item in data:
        new_item = {
            'question': item['prompt'],
            'answer': item['completion']
        }
        converted_data.append(new_item)
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

# 使用示例
input_file = '/cpfs/user/boyuan/workspace/aligner-reproduce/generate_scripts/test/glm-4-9b-chat_num_1_task_blank_model_glm-4.json'
output_file = input_file.replace('.json', '_converted.json')

try:
    convert_keys(input_file, output_file)
    print(f'Finished! New file saved as: {output_file}')
except Exception as e:
    print(f'Error: {str(e)}')
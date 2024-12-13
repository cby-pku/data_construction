from transformers import AutoTokenizer
from typing import Optional

# model template
# should contain system prompt and user prompt
# actually, you can also see the task template as the user prompt

PKU_ALPACA_MODEL_TEMPLATE: str = """BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"""
BAICHUAN_MODEL_TEMPLATE: str = """ <reserved_102> {prompt} <reserved_103> """




# task template
GAME24_TASK_TEMPLATE: str = """Please finish the following question game24 puzzle: {prompt}."""
TRANSLATE_TASK_TEMPLATE: str = """Please translate the following text from English to Chinese: {prompt}."""
CORRECTION_TASK_TEMPLATE: str = """Please correct the following answer to make it more safe and helpful: ##Question: {prompt} ##Response: {response} ##Your Correction:"""

BLANK_TEMPLATE: str = "{prompt}"





def get_task_template(task_name: str) -> str:
    template_dict = {
        'game24': GAME24_TASK_TEMPLATE,
        'blank': BLANK_TEMPLATE,
        'translate_en_to_zh': TRANSLATE_TASK_TEMPLATE,
        'correction': CORRECTION_TASK_TEMPLATE,
    }
    
    template = template_dict.get(task_name)
    
    if template is None:
        available_templates = ", ".join(template_dict.keys())
        raise ValueError(f"No task template found for task template {task_name}. You can choose from: {available_templates}")
        
    return template
    
    
def get_model_template(prompt: str, model_template: str, model_path: Optional[str] = None) -> str:
    
    template_dict = {
        'pku_alpaca': PKU_ALPACA_MODEL_TEMPLATE,
        'blank': BLANK_TEMPLATE,
        'baichuan': BAICHUAN_MODEL_TEMPLATE,
    }
    
    print(f"model_template: {model_template}")
    
    if model_template == 'qwen2':
        
        if model_path is None:
            raise ValueError("model_path is required for qwen2")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        
    elif model_template == 'llama3-ch' or model_template == 'glm-4' or model_template == 'yi':
        if model_path is None:
            raise ValueError("model_path is required for llama3-ch or chatglm-4 or yi")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        chat = [
            {"role": "user", "content": prompt},
        ]   
        input_ = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, trust_remote_code=True
        )
        return input_

    
    
    template = template_dict.get(model_template)
    
    if template is None:
        available_templates = ", ".join(template_dict.keys())
        raise ValueError(f"No model template found for model {model_template}. You can choose from: {available_templates}")
    
    return template

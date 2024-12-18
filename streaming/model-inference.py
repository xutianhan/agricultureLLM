from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载微调后的ChatGLM模型
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 进行推理测试
input_text = "如何种植玉米？"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(inputs['input_ids'], max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

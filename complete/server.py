from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型和分词器，注意指定正确的配置
model_name = "THUDM/chatglm3-6b"  # 使用您自己微调的模型路径
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)  # 使用trust_remote_code参数
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

app = FastAPI()

class Item(BaseModel):
    input_text: str

# 生成完整的模型输出
def generate_response(input_text: str):
    # 对输入进行编码
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 生成模型输出（您可以调整max_length等参数来控制输出的长度）
    outputs = model.generate(inputs["input_ids"], max_length=1000, do_sample=True, temperature=0.7)

    # 解码生成的输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/predict/")
async def predict(item: Item):
    response = generate_response(item.input_text)
    return {"response": response}  # 返回一个包含生成内容的字典

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
import time
import torch

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型和分词器，注意指定正确的配置
model_name = "THUDM/chatglm3-6b"  # 使用您自己微调的模型路径
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

app = FastAPI()

class Item(BaseModel):
    input_text: str

# 逐字流式生成并拼接响应
def generate_response(input_text: str):
    # 对输入进行编码
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 生成模型输出（您可以调整max_length等参数来控制输出的长度）
    outputs = model.generate(inputs["input_ids"], max_length=1000, do_sample=True, temperature=0.7, return_dict_in_generate=True, output_scores=True)

    # 初始部分内容为空
    full_response = ""

    # 逐字生成并拼接
    for idx in range(1, len(outputs.sequences[0]) + 1):
        partial_output = tokenizer.decode(outputs.sequences[0][:idx], skip_special_tokens=True)
        full_response = partial_output  # 将新的部分拼接到之前的内容
        yield full_response + "\n"  # 每次返回当前完整的输出内容
        time.sleep(0.1)  # 延迟以模拟逐字流式输出的效果

@app.post("/predict/")
async def predict(item: Item):
    # 获取逐字流式响应
    return StreamingResponse(generate_response(item.input_text), media_type="text/plain")
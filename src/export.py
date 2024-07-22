from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/llamac",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# {}
# ### Input:
# {}
# ### Response:
# {}"""
# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "他们在公里玩得很开心。",
#         "",
#         "",
#     )
# ], return_tensors = "pt").to("cuda")
#
# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
# print(_)

# 8ビットQ8_0に保存
if True:
    model.save_pretrained_gguf("model", tokenizer)
# https://huggingface.co/settings/tokens にアクセスしてトークンを取得してください。
# また、hfを自分のユーザー名に変更してください。
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# 16ビットGGUFに保存
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# q4_k_m GGUFに保存
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")

# 複数のGGUFオプションに保存 - 複数必要な場合ははるかに高速です！
if False:
    model.push_to_hub_gguf(
        "hf/model",  # hfを自分のユーザー名に変更してください。
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
        token="",  # https://huggingface.co/settings/tokens からトークンを取得してください
    )

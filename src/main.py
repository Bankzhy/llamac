from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/llama-3-8b-bnb-4bit",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     token = "https://hf-mirror.com"
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/llamac",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "https://hf-mirror.com"
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Please check the grammar of following sentence and fix it. If it does not have any error return 'True', else return the correct sentence only.",
        "Besides some technologically determinists that allow the development of biometric identification, this technology is also shaped by three social factors, namely, the desire of the society for safety, convenience and economy.",
        "",
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, max_new_tokens = 128)

result = tokenizer.batch_decode(_)
print(result)
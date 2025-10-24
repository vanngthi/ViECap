# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese')
# model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese')

# text = "Cậu bé đang chơi "
# input_ids = tokenizer.encode(text, return_tensors='pt')
# max_length = 100

# sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id,
#                                    do_sample=True,
#                                    max_length=max_length,
#                                    min_length=max_length,
#                                    top_k=40,
#                                    num_beams=5,
#                                    early_stopping=True,
#                                    no_repeat_ngram_size=2,
#                                    num_return_sequences=3)

# for i, sample_output in enumerate(sample_outputs):
#     print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))
#     print('\n---')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news")
model = AutoModelForCausalLM.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news", low_cpu_mem_usage=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

prompt = "Tiềm năng của trí tuệ nhân tạo" # your input sentence
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

gen_tokens = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=20,
    )

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

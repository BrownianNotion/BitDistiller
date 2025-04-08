import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_path = "../../models/Llama_3bit/" #but MUST have a working config & rope
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained("base-llama-or-whatever")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
enc = tokenizer(text, return_tensors="pt")
input_ids = enc["input_ids"].squeeze(0).cuda()

seqlen = 2048
loss_fct = nn.CrossEntropyLoss()

total_nll = 0.0
total_tokens = 0

for i in range(0, input_ids.size(0), seqlen):
    chunk = input_ids[i : i + seqlen]
    if chunk.size(0) < 2:
        break  # too short
    # Make a batch dimension [1, seq]
    chunk = chunk.unsqueeze(0)

    # 1) standard forward pass
    with torch.no_grad():
        outputs = model(chunk)
        # outputs.logits shape: [1, seqlen, vocab_size]

    # 2) shift for cross-entropy
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = chunk[:, 1:].contiguous()
    # compute cross-entropy
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    n_tokens = labels.numel()
    total_nll += loss.item() * n_tokens
    total_tokens += n_tokens

ppl = torch.exp(torch.tensor(total_nll / total_tokens))
print("Perplexity:", ppl.item())


# Hugging face Links: 

https://huggingface.co/fredericowieser

https://huggingface.co/BrownianNotion

https://huggingface.co/Heisenger

https://huggingface.co/acoleman

https://huggingface.co/VictorFiz


# Tiny Llama Model Results: 

### TinyLlama 3bit

https://huggingface.co/fredericowieser/TinyLlama_v1.1_3bit_int_3_bit_int

| PPL   | arc_easy         | arc_challenge     | piqa            | winogrande       | hellaswag        | mmlu | QA Avg |
|-------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 11.99 | 41.96 ± 1.01     | 24.32 ± 1.25      | 66.21 ± 1.10     | 50.28 ± 1.41     | 39.00 ± 0.49     | -    | 44.35  |

## TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline

https://huggingface.co/BrownianNotion/TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline


| Metric                                | Dataset       | Split        | Source        | Value  |
|---------------------------------------|---------------|--------------|---------------|--------|
| Accuracy                              | ARC-Challenge | Test Set     | Self-reported | 0.215  |
| Normalized Accuracy                   | ARC-Challenge | Test Set     | Self-reported | 0.246  |
| Accuracy                              | HellaSwag     | Test Set     | Self-reported | 0.324  |
| Normalized Accuracy                   | HellaSwag     | Test Set     | Self-reported | 0.373  |
| Accuracy                              | PIQA          | Validation   | Self-reported | 0.608  |
| Normalized Accuracy                   | PIQA          | Validation   | Self-reported | 0.607  |
| Accuracy                              | Winogrande    | Test Set     | Self-reported | 0.520  |
| QA Average                            | QA-Avg        | -            | Self-reported | 0.417  |
| Perplexity                            | WikiText-2    | Test Set     | Self-reported | 22.655 |



## TinyLlama_v1.1_mix_wikitext_alpaca_1bit_BitDistiller_baseline

https://huggingface.co/fredericowieser/TinyLlama_v1.1_mix_wikitext_alpaca_1bit_BitDistiller_baseline


| PPL  | arc_easy        | arc_challenge     | piqa           | winogrande      | hellaswag       | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| n/a | n/a              | 24.7              | 52.8            | 51.2                |    25.7      | -    | 37.8  |


---


## TinyLlama_v1.1_2bit_int_3x_data_3_epochs

https://huggingface.co/BrownianNotion/TinyLlama_v1.1_2bit_int_3x_data_3_epochs

| PPL      | arc_easy        | arc_challenge    | piqa            | winogrande       | hellaswag       | mmlu | QA Avg |
|----------|------------------|------------------|------------------|------------------|------------------|------|--------|
| 17899.79 | 26.26 ± 0.90     | 22.01 ± 1.21     | 52.01 ± 1.17     | 51.22 ± 1.40     | 25.96 ± 0.44     | -    | 35.49  |



# Changing Teacher Size: 


### TinyLlama 2bit (7B teacher)

https://huggingface.co/VictorFiz/TinyLlama_v1.1_2bit_int_llama2_7b_teacher


| PPL   | arc_easy         | arc_challenge     | piqa            | winogrande       | hellaswag        | mmlu | QA Avg |
|-------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 16.94 | 36.91 ± 1.00     | 20.14 ± 1.00      | 60.28 ± 1.00     | 53.99 ± 1.00     | 33.00 ± 1.00     | -    | 40.86  |

---

# Changing Model Size: 

### Llama-2-7b-hf_1bit_int

https://huggingface.co/Heisenger/Llama-2-7b-hf_1bit_int


| PPL     | arc_easy        | arc_challenge     | piqa            | winogrande       | hellaswag        | mmlu | QA Avg |
|---------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 4082.93 | 26.26 ± 0.90     | 22.44 ± 1.22      | 52.45 ± 1.17     | 51.78 ± 1.40     | 25.88 ± 0.44     | -    | 35.76  |

---

### Llama-2-7b-hf_2bit_int

https://huggingface.co/BrownianNotion/Llama-2-7b-hf_2bit_int

| PPL  | arc_easy        | arc_challenge     | piqa           | winogrande      | hellaswag       | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 7.87 | 67.09 ± 0.96     | 33.02 ± 1.37      | 74.05 ± 1.02     | 61.64 ± 1.37     | 48.79 ± 0.50     | -    | 56.92  |

---

## Llama-3.2-3B_2bit_int 

https://huggingface.co/acoleman/Llama-3.2-3B_2bit_int


PPL: 16.895

| Dataset        | Accuracy | Accuracy StdErr | Normalized Accuracy | Normalized Accuracy StdErr |
|----------------|----------|------------------|----------------------|-----------------------------|
| ARC Easy       | 0.5644   | ± 0.0102         | 0.5130               | ± 0.0103                    |
| ARC Challenge  | 0.2739   | ± 0.0130         | 0.2986               | ± 0.0134                    |
| HellaSwag      | 0.3976   | ± 0.0049         | 0.5062               | ± 0.0050                    |
| PIQA           | 0.6882   | ± 0.0108         | 0.6970               | ± 0.0107                    |
| Winogrande     | 0.5430   | ± 0.0140         | N/A                  | N/A                         |

QA avg: 0.4757

---

## Llama-3.2-3B_3bit_int

https://huggingface.co/acoleman/Llama-3.2-3B_3bit_int

PPL: 11.585

| Dataset        | Accuracy | Accuracy StdErr | Normalized Accuracy | Normalized Accuracy StdErr |
|----------------|----------|------------------|----------------------|-----------------------------|
| ARC Challenge  | 0.3669   | ± 0.0141         | 0.4019               | ± 0.0143                    |
| ARC Easy       | 0.6923   | ± 0.0095         | 0.6793               | ± 0.0096                    |
| HellaSwag      | 0.4945   | ± 0.0050         | 0.6565               | ± 0.0047                    |
| PIQA           | 0.7492   | ± 0.0101         | 0.7601               | ± 0.0100                    |
| Winogrande     | 0.6511   | ± 0.0134         | N/A                  | N/A                         |

QA avg: 0.5908


# Ablations: 


### Impact of Group Size on Performance

https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g32
https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g64
https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g128

| Group Size | PPL     | ARC-Challenge | ARC-Easy | PIQA   | Winogrande | QA Avg |
|------------|---------|----------------|----------|--------|-------------|--------|
| 128        | 3488.38 | 21.25%         | 26.56%   | 52.77% | 49.96%      | 37.63% |
| 64         | 2707.56 | 20.56%         | 25.76%   | 53.05% | 49.57%      | 37.23% |
| 32         | 2720.33 | 22.10%         | 24.75%   | 53.26% | 51.07%      | 37.79% |


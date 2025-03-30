
# Hugging face Links: 

https://huggingface.co/fredericowieser

https://huggingface.co/BrownianNotion

https://huggingface.co/Heisenger

https://huggingface.co/acoleman

https://huggingface.co/VictorFiz


# Tiny Llama Model Results: 

### 3bit

https://huggingface.co/fredericowieser/TinyLlama_v1.1_3bit_int_3_bit_int

| PPL   | arc_easy         | arc_challenge     | piqa            | winogrande       | hellaswag        | mmlu | QA Avg |
|-------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 11.99 | 41.96 ± 1.01     | 24.32 ± 1.25      | 66.21 ± 1.10     | 50.28 ± 1.41     | 39.00 ± 0.49     | -    | 44.35  |

### 3bit nf3 quantisation
https://huggingface.co/fredericowieser/TinyLlama_v1.1_3bit_nf3

| PPL |  arc_easy  |arc_challenge|    piqa    | winogrande | hellaswag  |mmlu|QA Avg|
|----:|------------|-------------|------------|------------|------------|----|-----:|
|11.52|43.60 ± 1.02|23.46 ± 1.24 |66.38 ± 1.10|52.01 ± 1.40|38.80 ± 0.49|-   | 44.85|

## 2bit

https://huggingface.co/BrownianNotion/TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline


| PPL | arc_easy | arc_challenge | piqa | winogrande | hellaswag | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
|  23.76  | 35.90 ± 0.98 | 22.10 ± 1.21 | 60.45 ± 1.14 | 52.88 ± 1.40 | 32.35 ± 0.48 | - | 40.71 |

## 1bit 

https://huggingface.co/fredericowieser/TinyLlama_v1.1_mix_wikitext_alpaca_1bit_BitDistiller_baseline


| PPL  | arc_easy        | arc_challenge     | piqa           | winogrande      | hellaswag       | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 2297.22 | 26.64 ± 0.91              | 21.50 ± 1.20             | 52.83 ± 1.16            | 51.22 ± 1.40                |    25.68 ± 0.44      | -    | 35.58  |


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


| PPL | arc_easy | arc_challenge | piqa | winogrande | hellaswag | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 16.895 | 56.44 ± 1.02 | 27.39 ± 1.30 | 68.82 ± 1.08 | 54.30 ± 1.40 | 39.76 ± 0.49 | - | 47.57 |

---

## Llama-3.2-3B_3bit_int

https://huggingface.co/acoleman/Llama-3.2-3B_3bit_int

| PPL | arc_easy | arc_challenge | piqa | winogrande | hellaswag | mmlu | QA Avg |
|------|------------------|-------------------|------------------|------------------|------------------|------|--------|
| 11.585 | 69.23 ± 0.95 | 36.69 ± 1.41 | 74.92 ± 1.01 | 65.11 ± 1.34 | 49.45 ± 0.50 | - | 59.08 |


# Ablations: 

## Different Loss function
### TinyLlama_v1.1_2bit_int_ce_plus_cakld_20
https://huggingface.co/Niks898/TinyLlama_v1.1_2bit_int_ce_plus_cakld_20

|     PPL      |  arc_easy  |arc_challenge|    piqa    | winogrande | hellaswag  |mmlu|QA Avg|
|--------------|------------|-------------|------------|------------|------------|----|-----:|
|2242.81 |35.40 ± 0.98|21.67 ± 1.20 |60.55 ± 1.14|51.93 ± 1.40|32.22 ± 0.47|-   | 40.36|

### Impact of Group Size on Performance

https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g32
https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g64
https://huggingface.co/BrownianNotion/tinyllama_v1.1-int1-g128

| Group Size | PPL     | ARC-Challenge | ARC-Easy | PIQA   | Winogrande | QA Avg |
|------------|---------|----------------|----------|--------|-------------|--------|
| 128        | 3488.38 | 21.25         | 26.56   | 52.77 | 49.96      | 37.63 |
| 64         | 2707.56 | 20.56         | 25.76   | 53.05 | 49.57      | 37.23 |
| 32         | 2720.33 | 22.10         | 24.75   | 53.26 | 51.07      | 37.79 |


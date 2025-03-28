
FBI-LLM:
* Columnwise scaling, learns alpha, beta linear scale + shift factors. Otherwise just signs the column weights, quite simple
* Only teacher/student loss, cross entropy instead of KL. No CE term.
* Use pretrained model as teacher (eg. Llama2-7B), also the full-precision version of student like in FBI LLM.
* Found that training from scratch was more suited to 1B than pretrained LLm weight (seems consistent with bitnet) 
* Is essentially training from scratch, even with distillation. Uses pajama dataset, massive number of tokens.

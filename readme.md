# BitDistiller-Extensions
Forked from the BitDistiller [paper](http://arxiv.org/abs/2402.10631) repo https://github.com/DD-DuDa/BitDistiller.git. Please cite the original repo if you find this work interesting.

```
@misc{du2024bitdistiller,
      title={BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation}, 
      author={Dayou Du and Yijia Zhang and Shijie Cao and Jiaqi Guo and Ting Cao and Xiaowen Chu and Ningyi Xu},
      year={2024},
      eprint={2402.10631},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
This is a student project with the goal being to explore unanswered questions from the BitDistiller paper such as:
- How does the approach do on smaller models (eg. TinyLlama 1.1B)?
- Does this approach work for 1/1.58 bit quantisation?
- How does the choice of teacher model affect performance?

## Contents
0. [Summary](#0-overall-workflow-summary)
1. [Setup](#1-setup)
2. [Pre-Training](#2-pre-training)
3. [Training workflow](#3-training-workflow)
4. [Uploading the Model](#4-uploading-the-model-to-hugging-face)
5. [Eval](#5-eval)

## 0. Overall Workflow Summary
0. Create new branch, clone repo on a cloud GPU instance. 
1. Run [Setup](#1-setup)
2. Run [Pre-Training](#2-pre-training) if applicable
3. Run [Training](#3-training-workflow) 
4. [Upload model](#4-uploading-the-model-to-hugging-face) to hugging face.
5. Run [Eval](#5-eval) to generate metrics.
6. **Delete instance!!**


## 1. Setup
### Logging into a cloud GPU with ssh 
If you haven't already done so on your local machine, do the steps [below](#generate-an-ssh-key-and-add-it-to-github) so that you can clone, pull, push etc. locally.
#### Generate an ssh key
```
eval "$(ssh-agent -s)" # start ssh agent, not automatic on vast
ssh-keygen -t ed25519
ssh-add; ssh-add -l
echo "public key:"
cat ~/.ssh/id_ed25519.pub
```
Press enter when prompted for file name/passphrase to use defaults. Copy the entire public key (including ssh-ed25519 and your email at the end) and add this to github under Settings > ssh keys.

#### Logging into instance
Add your local ssh key to your cloud GPU platform eg. Lambdalabs or vast.ai and create an instance with CUDA version 12.4. Login via vscode's remote ssh extension using 
```
ssh -i ~/.ssh/id_ed25519 -p port user@address # (+optional port forwarding with -L)
```
eg. on vast
```
ssh -i ~/.ssh/id_ed25519 -p 30077 root@185.150.27.254 -L 8080:localhost:8080
```

#### Working on your instance
Repeat the setps in [Generate an ssh key](#generate-an-ssh-key-and-add-it-to-github) on your **remote instance** and clone the repo.
```
git clone git@github.com:BrownianNotion/BitDistiller.git
```

### Setting up the python environment
Run `./setup.sh` to setup env/install packages. Activate the venv with
```
source BitDistillerVenv/bin/activate
```
Note that for vast.ai, your repo will be under `/workspace/BitDistiller`.

## 2. Pre-Training
With all steps, change the output paths (eg. for clipped weights, checkpoints) to match
the name of your experiment.

### Clipping
Clips/quantises the teacher model (eg. TinyLlama_v1.1 below) to get initial weights for quantised student model. **Shouldn't need to be rerun unless using a new teacher/quantisation method**. Initial weights stored in `--dump_clip` argument.
```
cd quantization

CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path ../models/TinyLlama_v1.1 --calib_dataset pile --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/TinyLlama_v1.1/int2-g128.pt
```

### Generate Teacher Data
Generate the data for (distillation) training. **Shouldn't need to be rerun unless using a new teacher**. The main file we will use for training is `data/datasets/tinyllama_v1.1/mix_wiki_alpaca_8000.json`.
```
cd data/generation

bash generate.sh ../../models/TinyLlama_v1.1 wikitext ../datasets/tinyllama_v1.1/ 16 3000
bash generate.sh ../../models/TinyLlama_v1.1 alpaca ../datasets/tinyllama_v1.1/ 16 5000

# change to path in .py
python mix_data.py
```

## 3. Training workflow
The model is by default trained on the dataset `mix_wiki_alpaca_8000.json`. Make sure to change the `bits`, `quant_type`, and `--clip` (initial clipped weights) path and any other training parameters needed in `train.sh`. If doing a dry-run, change the parameters in`train_dry_run.sh` instead. 

### Summary of steps
1. Commit **all** changes made by your experiment to a branch for reproducibility. This includes changes to `train.sh` and other configs other than dry run. 
2. Rerun clipping/data generation if needed (see [Pre-Training](#2-pre-training)).
3. In `train/`, change `train_dry_run.sh` if needed and run `./dry_sun.sh` to check that your code works. This does a single step on a small dataset of 64 samples.
4. (Skip if on vast.ai) If dry run succeeds, create a new tmux session:
```
tmux new -s session_name
```
If your ssh connection ever drops, your training will keep running. You may need to reattach your session.
```
tmux attach -t session_name
```
5. Run the training command below. Once the model starts training, see [Monitoring](#monitoring) below for how to monitor training.
```
cd train
bash train.sh ../data/datasets/tinyllama_v1.1/mix_wiki_alpaca_8000.json ./ckpts/tinyllama_v1.1/int2-g128/ ./ckpts/tinyllama_v1.1/int2-g128/runs/ 4
```

### Monitoring
Run these commands in new terminals once actual training has started (i.e. you see two progress bars).
```
source BitDistillerVenv/bin/activate
cd train

# Nice dashboard of train/validation loss and other metrics. Eval metrics won't appear
# until an eval step has happened - this may take a while.
tensorboard --logdir=ckpts/tinyllama_v1.1/int2-g128/runs/ --port=8008

# (In new terminal)
# Shows GPU and GPU memory usage. This should be close to 100%/36.5GB for training.
nvtop
```

Signs your training has gone wrong (to be expanded):
* The loss curve isn't going down after a few steps

## 4. Uploading the model to hugging face 
As eval takes time, begin uploading the model as soon as training has finished if the loss curves and validation metrics look good.

### Logging into hugging face
Login to hugging face with your access token (generate one if you don't have one) with
```
huggingface-cli login
```
Check your login succeeded with
```
huggingface-cli whoami
```

### Uploading the model
Make sure your tensorboard logs (`.events.out.tfevents.{...}`) are inside your `<model_path>` folder (hugging face will auto-generate a [metrics tabs](https://huggingface.co/docs/hub/en/tensorboard) to display the loss curves). 

Run `upload_model.py`, specifying args `<model_path>`, `<bits>` and optionally
`--quant_type, --extra_changes, --base_model, --ovewrite`. Run `upload_model.py -h` for help on the options. 

This uploads the model to the hugging face repo `your_username/model_name`. Model name follows the convention
*"{base_model}\_{num}bit\_{quantisation method}(\_{extra changes})"*.

**Example Usage**
```
python upload_model.py train/ckpts/tinyllama_v1.1/int2-g128/checkpoint-100 2 --quant_type int --extra_changes ce_loss 
```

## 5. Eval
Our main benchmarks will be perplexity (PPL), QA datasets (arc_easy, arc_challenge, winogrande, hellasawg, piqa) and MMLU. For consistency, do not change `num_fewshot`. These benchmarks can be run as follows:
```
cd test/general

# PPL
python wiki_ppl.py --model ../../train/ckpts/tiny_llama_v1.1/int2-g128/checkpoint-12/ --quant_type int --bits 2 --group_size 128

# QA
CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model ../../train/ckpts/tinyllama_v1.1/int2-g128/checkpoint-12/ --eval_tasks arc_easy,arc_challenge,winogrande,hellaswag,piqa --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 0 

# MMLU
CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model  ../../train/ckpts/tinyllama_v1.1/int2-g128/checkpoint-12/ --eval_tasks hendrycksTest-* --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 5
```

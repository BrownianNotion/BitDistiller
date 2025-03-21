* Note to self: before trying big boy GPUs like h100s to run things faster, make sure things actually work on a cheap GPU...
* Get transformers error:
```
ValueError: --load_best_model_at_end requires the saving steps to be a multiple of the evaluation steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps 20 and eval_steps 0.1.
```
Curious that eval_steps=0.25 didn't seem to fail on Charlie's branch. Believe this might be a bug that was fixed in later versions of transformers https://discuss.huggingface.co/t/why-save-steps-should-be-a-round-multiple-of-eval-steps-when-load-best-model-at-end-true/10841/4.

* Need to change the `"best_model_checkpoint"` to in `trainer_state.json` to match the one specified in `resume_from_checkpoint` in hugging face trainer.

* Get 
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/BitDistiller/train/train.py", line 439, in <module>
[rank0]:     train()
[rank0]:   File "/workspace/BitDistiller/train/train.py", line 428, in train
[rank0]:     trainer.train(resume_from_checkpoint="ckpts/TinyLlama_v1.1_2bit_int_three_times_data/checkpoint-600")
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/transformers/trainer.py", line 1539, in train
[rank0]:     return inner_training_loop(
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/transformers/trainer.py", line 1825, in _inner_training_loop
[rank0]:     self._load_rng_state(resume_from_checkpoint)
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/transformers/trainer.py", line 2326, in _load_rng_state
[rank0]:     checkpoint_rng_state = torch.load(rng_file)
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/torch/serialization.py", line 1470, in load
[rank0]:     raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
[rank0]: _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
[rank0]:        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
[rank0]:        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
[rank0]:        WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([_reconstruct])` or the `torch.serialization.safe_globals([_reconstruct])` context manager to allowlist this global if you trust this class/function.
```

hacked around it by adding weights_only=False in transformers library, but should be resolved by downgrading torch or upgrading transformers. Probably should start using package manager with how many conflicts we've had...

can also save/load with latest pytorch https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/discussions/3

* when resuming from checkpoint, specify total number of epochs including those already elapsed. eg. 4 epoch if you want to train 2 more epochs on a model already trained for 2 epochs.
* when training from a checkpoint, changing train.sh won't matter, need to modify trainer_state.json (eg. for eval_steps). 

* eval step ~2:20 on h100 sxm4, ~1hr (estimate) for 2 epochs on 24000 samples dataset.

* GPU memory usage still at ~36GB when resuming from checkpoint, not sure if this is due to batch size or perhaps there was some part in config I forgot to change.
<!-- * `snapshot_download` in `download_model.py` didn't seem to download the repo with checkpoint data, only a `pytorch_model.bin`. Not sure why. Ended up just cloning the hf repo instead. -->
* up to checkpoint 1080, best model was still 900. Killed early and decided to use this.
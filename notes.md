* Get transformers error:
```
ValueError: --load_best_model_at_end requires the saving steps to be a multiple of the evaluation steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps 20 and eval_steps 0.1.
```
Curious that eval_steps=0.25 didn't seem to fail on Charlie's branch. Believe this might be a bug that was fixed in later versions of transformers https://discuss.huggingface.co/t/why-save-steps-should-be-a-round-multiple-of-eval-steps-when-load-best-model-at-end-true/10841/4.

* Need to change the `"best_model_checkpoint"` to in `trainer_state.json` to match the one specified in `resume_from_checkpoint` in hugging face trainer.

<!-- * `snapshot_download` in `download_model.py` didn't seem to download the repo with checkpoint data, only a `pytorch_model.bin`. Not sure why. Ended up just cloning the hf repo instead. -->
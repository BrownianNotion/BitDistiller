## Notes on run

### Clipping
Before clipping, make sure to run 
'''
pip install --upgrade transformers
'''
for Llama3 models to handle rope_scaling in config.json.

###
* Clipping, example loss magnitude, a couple of orders higher than tinyllama 1.1B
loss: 0.00147247314453125
loss: 0.00012493133544921875
loss: 0.00125885009765625
loss: 0.00115966796875
loss: 0.00029754638671875
#### Runtime
* clipping ~20 min
* data gen ~20 min 
* training ~5h 40min

### Train notes
#### New errors
When running bash dry_run.sh, run into this error:
```
TOKENIZERS_PARALLELISM=(true | false)
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using tokenizers before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
[rank0]:[W323 08:31:49.969131077 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```
To overcome this, run 
```
export TOKENIZERS_PARALLELISM=false
```
in terminal. However, this then leads to an Out Of Memory error so batch size must be reduced. Moreover, once dry_run.sh run again, a new error occurs:
```
[rank0]: Traceback (most recent call last):
[rank0]:     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/transformers/trainer.py", line 3718, in training_step
[rank0]:     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
[rank0]: TypeError: compute_loss() got an unexpected keyword argument 'num_items_in_batch'
  0%|                                                                                                                                              | 0/1 [00:00<?, ?it/s]
[rank0]:[W323 08:38:20.172772860 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[2025-03-23 08:38:30,122] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 8176
[2025-03-23 08:38:30,123] [ERROR] [launch.py:325:sigkill_handler]
```
Add the argument 'num_items_in_batch' to KDTrainer. This leads to yet another Out Of Memory error, so decrease batch size to 4. Training now appears to complete, but get this error after final train step:

```
[rank0]:     return
[rank0]:   File "/workspace/BitDistiller/BitDistillerVenv/lib/python3.9/site-packages/torch/serialization.py", line 784, in __exit__
[rank0]:     self.file_like.write_end_of_file()
[rank0]: RuntimeError: [enforce fail at inline_container.cc:626] . unexpected pos 12851034048 vs 12851033880
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:28<00:00, 29.56s/it]
[rank0]:[W323 10:03:42.299538983 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[2025-03-23 10:03:55,561] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 4632
[2025-03-23 10:03:55,562] [ERROR] [launch.py:325:sigkill_handler] ['/workspace/BitDistiller/BitDistillerVenv/bin/python3.9', '-u', 'train.py', '--local_rank=0', '--model_name_or_path', '../models/Llama-3.2-3B/', '--data_path', '../data/datasets/Llama-3.2-3B/mix_wiki_alpaca_64.json', '--model_max_length', '1024', '--output_dir', './ckpts/dry_run', '--logging_dir', './logs/dry_run/', '--num_train_epochs', '1', '--bf16', 'True', '--seed', '42', '--per_device_train_batch_size', '4', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '4', '--gradient_checkpointing', 'True', '--evaluation_strategy', 'steps', '--eval_steps', '4', '--load_best_model_at_end', 'True', '--save_strategy', 'steps', '--save_steps', '20', '--save_total_limit', '3', '--learning_rate', '2e-5', '--lr_scheduler_type', 'constant', '--weight_decay', '0.', '--logging_steps', '1', '--report_to', 'tensorboard', '--deepspeed', 'config/zero.json', '--bits', '2', '--quant_type', 'int2-asym', '--q_group_size', '128', '--train_kd', 'True', '--kd_loss_type', 'cakld', '--max_train_samples', '999999', '--clip', '../quantization/clip_cache/Llama-3.2-3B/int2-g128.pt'] exits with return code = 1
```

#### Notes from BitDistiller Reproduction
```
du -sh BitDistiller/train/ckpts/Llama-2-7b-hf/int2-g128/checkpoint-200
```
checkpoint size: 101G
GPU memory usage hovering 60/80GB
optimiser state is 75GB, can't be uploaded to hugging face as it exceeds max individual size. If we need optimiser state, consider alternate solution, eg. is it possible to chunk the file

### Eval Notes
PPL 7.872644901275635, ~20s, very quick
eval only 15 min very quick, maybe due to internet speed rather than gpu?

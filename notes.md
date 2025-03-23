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
* clipping ~40 min
* data gen ~35 min 
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


```
du -sh BitDistiller/train/ckpts/Llama-2-7b-hf/int2-g128/checkpoint-200
```
checkpoint size: 101G
GPU memory usage hovering 60/80GB
optimiser state is 75GB, can't be uploaded to hugging face as it exceeds max individual size. If we need optimiser state, consider alternate solution, eg. is it possible to chunk the file

### Eval Notes
PPL 7.872644901275635, ~20s, very quick
eval only 15 min very quick, maybe due to internet speed rather than gpu?
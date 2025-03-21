## Notes on run
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
* training ~5h 40min (estimate based on halfway) 

### Train notes
```
du -sh BitDistiller/train/ckpts/Llama-2-7b-hf/int2-g128/checkpoint-200
```
checkpoint size: 101G
GPU memory usage hovering 60/80GB

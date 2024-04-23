## Self Rewarding Language Model

Implement for [Self-Rewarding Language Model paper](https://arxiv.org/abs/2401.10020) from MetaAI.

#### step 1: SFT (M0 -> M1)
```
python srlm/finetune/SFT.py
```
#### step 2: Generates dpo dataset 
```
python srlm/generate/Generate.py
```
#### step 3: DPO (M1 -> M2)
```
python srlm/finetune/DPO.py
```

#### step 4:(option) upload model
Use notebook/upload_model.ipynb to upload final_checkpoint model into huggingface hub.






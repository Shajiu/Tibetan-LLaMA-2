# Tibetan-LLaMA 2: Large Language Model for Tibetan (An Efficiently Extensible Large Language Model for Tibetan)

![Tibetan](https://github.com/Shajiu/Tibetan-Llama-2/blob/main/cpt_with_peft_lora/logo.png#pic_center=600x200)

**I. Introduction:**

- This project releases **Tibetan-LLaMA 2: Large Language Model for Tibetan**, aiming to further promote open research on large language models within the Tibetan NLP community. These models **extend the vocabulary of the original LLaMA** and undergo continued pre-training using Tibetan data, further enhancing their foundational semantic understanding of Tibetan. Meanwhile, the Tibetan Chat models are further fine-tuned using Tibetan instruction data, significantly improving their ability to understand and follow instructions.

**II. Abstract:**

- This study addresses the limitations of large language models in handling Tibetan language tasks and proposes a comprehensive solution.
- First, we construct a large-scale, high-quality Tibetan unsupervised pre-training dataset containing 15 GB of data and a supervised multi-task fine-tuning dataset containing 1 million samples, effectively alleviating the scarcity of Tibetan language resources.
- Furthermore, by analyzing the unique morphological and grammatical characteristics of Tibetan, we select a Unigram-based tokenization strategy and develop an efficient Tibetan tokenizer.
- In addition, by extending the vocabulary of the Llama 2-7B/13B models and conducting mixed continued pre-training with Tibetan and English/Chinese data at a ratio of 3:4, we significantly enhance the models' Tibetan understanding and generation capabilities.
- To optimize training and deployment efficiency while improving the models' understanding of Tibetan semantics and instruction-following capabilities, we apply the LoRA+ fine-tuning method to a supervised multi-task instruction dataset, further improving model performance.
- To validate the effectiveness of the proposed methods, we establish ten Tibetan-oriented evaluation benchmarks and conduct comprehensive evaluations through both automatic and human assessments. The results demonstrate significant improvements in Tibetan understanding and generation while maintaining strong English capabilities. The key innovation of this study lies in systematically addressing the efficiency and performance challenges of LLMs when processing Tibetan data, particularly through a novel tokenization strategy and mixed pre-training approach that substantially improve processing efficiency and generalization capabilities.
- All research成果 have been open-sourced, providing valuable resources and a solid foundation for LLM applications and research in Tibetan and other minority languages. These contributions promote the research and deployment of LLMs for specific languages while making an important contribution to linguistic diversity in artificial intelligence.

**III. Contributions:**

- 🚀 Extend the Tibetan vocabulary of Llama 2 to improve encoding and decoding efficiency. Compared with the original Llama 2, the Tibetan sequence length is reduced by approximately 42.64%, and the maximum Tibetan text input length supported by the model is increased to more than twice its original capacity.
- 🚀 Conduct a comprehensive preliminary exploration of tokenization strategies for Tibetan and ultimately select Unigram as the optimal tokenization strategy, addressing the poor Tibetan tokenization performance of the original base model and enhancing its ability to understand and generate Tibetan content.
- 🚀 Perform mixed CPT using Tibetan and English/Chinese data at a ratio of 3:4, followed by Tibetan instruction fine-tuning. We also open-source the Base and Chat model weights based on Llama 2-7B/13B, supporting a broader range of applications and research.

**IV. Training Strategy:**

This study adopts a two-stage training approach:

- Stage 1: Freeze the parameters of the model's Transformer components and train only the Embedding layer, adapting the newly added Tibetan token embeddings while minimizing interference with the original model.
- Stage 2: Add LoRA+ weights to the model and train the LoRA+ parameters together with the Embedding layer. Although this two-stage training approach is relatively less efficient, it effectively alleviates the distribution shift encountered during CPT due to the discrepancy between the Tibetan data distribution and the data distribution used during the pre-training of Llama 2.

**V. Usage:**

#### 1. Environment Setup

```sh
pip install -r requirements.txt
```

#### 2. Format the Continued Pre-training Data as Follows

```sh
# Stored in ./data/data.txt, with one sample per line
གྲོང་ཚོར་བཅའ་སྡོད་བྱ་ཡུལ་དུ་ཁྲང་ཧྥུའུ་ལེའང་གིས་གྲོང་ཚོར་བཅའ་སྡོད་ལས་བྱེད་པ་དང་བསམ་བློ་བརྗེ་རེས་བྱས་པ་དང་། གཞུང་ལས་ཁང་(ཅུད)གི་ཏང་ཙུའུ་ཡིས་བཅའ་སྡོད་ལས་བྱེད་པར་ཐུགས་ཁུར་ཟབ་བཞེས་གནང་ལུགས་བརྒྱུད་བསྒྲགས་བྱས་པ་མ་ཟད། གྲོང་ཚོར་བཅའ་སྡོད་ལས་བྱེད་པའི་འཚོ་བ་དང་བདེ་ཐང་གི་གནས་ཚུལ་ལ་རྒྱུས་ལོན་བྱས་ཁར་། གྲོང་ཚོར་བཅའ་སྡོད་ལས་དོན་རུ་ཁག་གི་ལས་དོན་སྙན་ཞུ་ནན་ཏན་ངང་གསན་པ་དང་ཁོང་ཚོར་ཐོབ་པའི་གྲུབ་འབྲས་ལ་ཁས་ལེན་གང་ལེགས་གནང་བ་རེད།
```

#### 3. CPT Training

```sh
sh run_pt.sh
```

The detailed configuration is as follows:

```sh

lr=2e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=per_models/Llama-2-13b-hf                 # Local path to the downloaded Llama-2-13b-hf base model
tibetan_tokenizer_path=per_models/merged_tokenizer_hf      # Path to the expanded Tibetan vocabulary
dataset_dir=data                                           # Folder containing the training data
data_cache=temp_data_cache_dir       
per_device_train_batch_size=12
gradient_accumulation_steps=8
block_size=256
output_dir=output_dir_13

deepspeed_config_file=ds_z2_config.json                     # DeepSpeed configuration file

torchrun --nnodes 1 --nproc_per_node 1 --master_port=25678 train.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tibetan_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.005 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 100000 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```

#### 4. Merge Weights

```sh
sh run_merge_export.sh
```

The detailed commands are as follows:

```sh

# Convert to Hugging Face format
python3.9 merge_lora_low_mem.py \
    --base_model per_models/Llama-2-13b-hf \
    --lora_model output_dir_7B/checkpoint-219800 \
    --output_type huggingface \
    --output_dir ./merge_llama2_with_chinese_lora_13B/huggingface/ 

-----------------------------------------------------------------OR-----------------------------------------------------------------

# Convert to PTH format
python3.9 merge_lora_low_mem.py \
    --base_model per_models/Llama-2-13b-hf \
    --lora_model output_dir_7B/checkpoint-219800 \
    --output_type pth \
    --output_dir ./merge_llama2_with_chinese_lora_13B/pth/
```

#### 5. Start the Service

```sh
python serve_demo.py
```

#### 6. How to Send Service Requests

```sh
POST   http://10.x.x.x:8718/api/chat

{"query":"xxxxxxxxx"}
```

#### 7. Offline Interactive Chat / File Prediction

- Inference using the Transformers library

```sh
python inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive \
```

- Accelerated inference using vLLM

```sh
python inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive \
    --use_vllm
```

#### 8. Quantized Deployment Based on llama.cpp

- Install llama.cpp

```sh
https://github.com/ggerganov/llama.cpp
```

- Compile

```sh
make LLAMA_CUBLAS=1
```

- Convert the above .pth/bin model weights to the GGML FP16 format

```
python convert_hf_to_gguf.py ../path_to_llama3_chinese_instruct_hf_dir/
```

- Quantize the FP16 model to 4-bit

```
./llama-quantize.exe ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-f16.gguf ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.bin q4_0
```

- Interactive testing

```
./llama-cli.exe --conversation -m ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -c 2048 --temp 0.2 -n 256
```

- Server version

```sh
./llama-server.exe -m ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.gguf -c 2048
```

**V. Model Downloads:**

| Model Name | Type | Training Data | Reconstructed Model | Size | Download Link |
| :-----| ----: | :----: |:----: |:----: |:----: |
| Tibetan_Llama2_7B | Base | 8B | Original LLaMA-7B | 12.9 GB | [Tibetan_Llama2_7B](https://huggingface.co/shajiu/Yak_Llama2_7B) |
| Tibetan_Llama2_13B | Base | 13B | Original LLaMA-7B | 24.7 GB | [Tibetan_Llama2_13B](https://huggingface.co/shajiu/Yak_Llama2_13B) |


**VI. Dataset Downloads**

| Dataset Name | Data Type | Dataset Size | Download Link |
| :-----| ----: | :----: |:----: |
| TibetanGeneral corpus | Continued Pre-training | 15GB | [TibetanGeneral corpus](https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus) |
| TibetanSft corpus | Instruction Fine-tuning | 1GB | [TibetanSft corpus](https://huggingface.co/datasets/shajiu/TibetanSft_corpus) |

For data security considerations and to mitigate potential harmful impacts, we have chosen not to release the complete safety evaluation dataset. The publicly downloadable test set in this repository contains only a subset of the data. However, researchers conducting safety research may request access by emailing `shajiu_dn@163.com`. Upon approval, we will provide applicants with the complete dataset.

**Statement:** The dataset contains examples of harmful and policy-violating content, which do not represent the views or positions of our team.

**VII. Disclaimer:**

The resources associated with this project are intended solely for academic research and must not be used for commercial purposes. When using components involving third-party code, please strictly comply with the corresponding open-source licenses. The content generated by the model may be affected by factors such as model computation, randomness, and quantization precision loss. This project makes no guarantees regarding the accuracy of the generated content. This project assumes no legal responsibility for any content generated by the model, nor does it accept liability for any losses that may arise from the use of the relevant resources or generated outputs. This project is initiated and maintained by individuals and collaborators during their spare time; therefore, we cannot guarantee timely responses or solutions to reported issues.

**VIII. Acknowledgments:**

This project is primarily developed based on the following open-source project. We would like to express our gratitude to the project and its research and development contributors.

- [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

**IX. Citation:**

If you use the data, code, or models from this project, please cite this work.

```text
@article{sha2025tibetan,
  title={Tibetan-LLaMA 2: Large Language Model for Tibetan},
  author={Sha, Jiu and Zhu, Mengxiao and Feng, Chong and Ci, Jizhuoma},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  year={2025},
  publisher={ACM New York, NY}
}
```

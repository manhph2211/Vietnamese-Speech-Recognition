Vietnamese-Speech-Recognition
=====

# Introduction

In this repo, I focused on building end-to-end speech recognition pipeline using [Quartznet](https://arxiv.org/abs/1910.10261), [wav2vec2.0](https://arxiv.org/abs/2006.11477) and [CTC decoder](https://github.com/parlance/ctcdecode) supported by beam search algorithm as well as language model. 

# Setup 

## Datasets

Here I used [100h speech public dataset](https://institute.vinbigdata.org/events/vinbigdata-chia-se-100-gio-du-lieu-tieng-noi-cho-cong-dong/) of **Vinbigdata** , which is a small clean set of [VLSP2020 ASR competition](https://vlsp.org.vn/vlsp2020). Some infomation of this dataset can be found at `data/Data_Workspace.ipynb`. The data format I would use to train and evaluate is just like [LJSpeech](), so I create `data/custom.py` to customize the given dataset.

```
mkdir data/LJSpeech-1.1 
python data/custom.py # create data format for training quartnet & w2v2.0
```

And below is the folder that I used, note that `metadata.csv` has 2 columns, `file name` and `transcript`:

```
├───data
│   ├───LJSpeech-1.1
│   │   └───wavs
│   │   └───metadata.csv
│   └───vlsp2020_train_set_02
├───datasets
├───demo
├───models
│   └───quartznet
│       └───base
├───tools
└───utils
```

## Environment

You can create your environment and install the requirements file and note that **torch** should be installed based on your CUDA version. With **conda**:

```
cd Vietnamese-Speech-Recognition
conda create -n asr
conda activate asr
conda install --file requirements.txt
```

Also, you need to install **ctcdecode**:

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install . && cd ..
```

# Tools

## Training & Evaluation

For training the quartznet model, you can run:

```
python3 tools/train.py --config configs/config.yaml
```

And evaludate quartnet: 

```
python3 tools/evaluate.py --config configs/config.yaml
```

Or you wanna finetune wav2vec2.0 model from Vietnamese pretrained [w2v2.0](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h):

```
python3 tools/fintune_w2v.py
```

## Demo

This time, I provide small code with streamlit for asr demo, you can run:
```
streamlit run demo/app.py
```
![demo](https://github.com/manhph2211/Vietnamese-Speech-Recognition/blob/main/demo/assets/demo.gif)


# Results

I used wandb&tensorboard for logging results and antifacts during training, here are some visualizations after several epochs:

Quartznet             |  W2v 2.0
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/61444616/195522590-ae3267bf-0a15-4407-ab0f-4d1aca3b20d6.png)  |  ![](https://user-images.githubusercontent.com/61444616/197971160-14d44cb8-25f8-43d8-9071-16bc5ec59ca7.png)


# References

- Mainly based on [this implementation](https://github.com/oleges1/quartznet-pytorch)
- The [paper](https://arxiv.org/abs/1910.10261)
- Vietnamese ASR - [VietAI](https://github.com/vietai/ASR)
- Lightning-Flash [repo](https://github.com/Lightning-AI/lightning-flash)
- Tokenizer used from [youtokentome](https://github.com/VKCOM/YouTokenToMe)
- Language model [KenLM](https://github.com/kpu/kenlm)

Vietnamese-Speech-Recognition
=====

# Introduction

In this repo, I focused on building end-to-end speech recognition pipeline using [Quartznet]() and [CTC decoder]() supported by beam search algorithm as well as language model. 

# Setup 

## Datasets

Here I used [100h speech public dataset]() of **Vinbigdata** , which is a small clean set of [VLSP2020 ASR competition](). Some infomation of this dataset can be found at `data/Data_Workspace.ipynb`. The data format I would use to train and evaluate is just like [LJSpeech](), so I create `data/custom.py` to customize the given dataset.

```
mkdir data/LJSpeech-1.1 
python data/custom.py
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
cd ctcdecode && pip install .
cd ..
```

# Tools

## Training & Evaluation

For training the quartznet model, you can run:

```
python3 tools/train.py --config configs/config.yaml
```

And evaludate: 

```
python3 tools/evaluate.py --config configs/config.yaml
```

## Inference

# Results

I used wandb for logging results and antifacts during training, here are some visualizations after several epochs:


# References


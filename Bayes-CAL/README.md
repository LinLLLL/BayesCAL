# Bayes-CAL

The official implementation of  **Bayesian Cross-modal Alignment Learning for Few-Shot Out-of-Distribution Generalization.**

We propose a novel Bayesian cross-modal alignment learn- ing method (Bayes-CAL) for few-shot OoD generalization. Unlike CoCoOp and DPLCLIP that fine-tune task-specific parameters by incorporating the conditional information ex- tracted from image features, we fine-tune on the semantic space by enforcing domain-invariant alignment under the proposed regularization terms. Moreover, the Bayesian treat- ment is specially introduced to substantially alleviate over- fitting. Based on the domain-invariant information disentan- gled from the image features, the distributions of the task- specific parameters are estimated. Without a query of a large amount of GPU memory like CoCoOp in every run, the pro- posed Bayes-CAL is simple yet efficient, making fine-tuning on few-shot samples practical in the few-shot OoD setting.



## How to Install

This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). 



## DATA Download

Most of the datasets, including MNIST (the base of ColoredMNIST), PACS, VLCS, NICO, OfficeHome can be downloaded publicly available but need to be downloaded manually.

The ColoredCatsDogs dataset  (CCD for short) has spurious correlations with the background color (green or red), constructed in a similar principle as ColoredMNIST but with images of cats and dogs disturbed by Gaussian noise to increase complexity. Here are some examples of the CCD.

The CCD dataset can be downloaded [here](https://pan.baidu.com/s/1za8Cp8PJyWWStTj88D4jGA?pwd=vjgf ).

<img src="Bayes-CAL-main/Figures/CCD.png" alt="CCD" style="zoom:4%;" />



Make sure that the directory structure of each dataset is arranged as follows:

#### MNIST

```
MNIST
└── processed
    ├── training.pt
    └── test.pt
```

#### ColoredMNIST

```
ColoredMNIST
├── train_1
├── train_2
├── val
└── test
```

#### ColoredCatsDogs

```
ColoredCatsDogs
├── train_1
├── train_2
├── val
├── test
└── ColoredCatsDogs.json
    
```

#### PACS

```
PACS
├── art_painting
├── cartoon
├── photo
└── sketch
```

#### VLCS

```
VLCS
├── art_painting
├── cartoon
├── photo
└── sketch
```

#### OfficeHome

```
office_home
├── Art
├── Clipart
├── Product
└── Real World
```

#### NICO

```
NICO
├── animal
├── vehicle
└── mixed_split_corrected
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```



## How to Run

We provide the running scripts in `scripts/`. Make sure you change the path in `DATA` and run the commands under `Bayes-CAL/scripts/`.



## Pipline

![pipline-a](Bayes-CAL-main/Figures/pipline-a.png)



## Collect Results

Run python collect_result.py to get the final results of after hyperparameters search based on the training-domain validation or test domain validation.



## Results

All model weights can de downloaded via this link. Run python collect_result.py, and you will get the following results.

![main_result](/Users/zl/Desktop/Bayes-CAL-main/Figures/main_result.jpg)

![base_to_new](/Users/zl/Desktop/Bayes-CAL-main/Figures/base_to_new.jpg)

## Loss Landscape

Set the max epoch at 50 to obtain the loss landscape.

![landscape](Bayes-CAL-main/Figures/landscape.png)

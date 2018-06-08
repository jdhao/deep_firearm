# Introduction

This is the code our ICPR2018 paper "DeepFirearm: Learning Discriminative Feature
Representation for Fine-grained Firearm Retrieval".


# How to run this code

## Package info

The code is written using the PyTorch version 0.3.0. So In order to run this code, you may
install version 0.3.0 of PyTorch or adapt it to the newer version of PyTorch.

## Instructions for dataset download

You can download the dataset from [here](http://forensics.idealtest.org/Firearm14k/). Two separate data
are used for the experiment. One is for classification training, and the other is for the retrieva
training. After downloading this dataset, extract it under the folder `data` using the following command:

```
tar -zxvf firearm-train-val.tar.gz -C data/ # for the classification data
```

```
tar -zxvf firearm-dataset.tar.gz -C data/ # for the retrieval data
```

## Train the classification model

In order to train the classification model, run the following command:

```
python train_cls.py
```

## Train the retrieval model after classfication

To get better retrieval performance, we further train the model using retrieval task based on the classification
model. To train the model, use the following command:

```
python train_retr_from_cls.py
```

## Benchmark on test set

To check the model's performance on test set, run the following command:

```
python benchmark_on_test.py
```

It will show both the mAP and rank-k accuracy for different feature dimensions.

# Citation information

The citation information for this code is:

```
@conference{Hao2018DFLD,
  title={{DeepFirearm}: Learning Discriminative Feature Representation for Fine-grained Firearm Retrieval},
  author={Jiedong Hao and Jing Dong and Wei Wang and Tieniu Tan},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  year={2018},
  publisher={IEEE Computer Society},
  address={Washington}
}
```
This repository includes the source code for the paper entitled: "MFANet: A Lightweight Multi-Feature Attention Network for Medical Image Classification".

## Dependencies
* Tensorflow = 2.10.0
* scikit-learn = 1.3.2
## Dataset (ISIC 2018 and Messidor)
Please download the ISIC 2018 using this  [link](https://challenge.isic-archive.com/data/#2018) , Pap Smear dataset using this [link](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed), and Breakhis dataset using this [link](https://www.kaggle.com/datasets/ambarish/breakhis).

## Data Preparing
To divide the dataset into the aprequired no. of clients, divide the data into training, validation and testing set as mentioned in the paper, then run non_iid_data_preparation.py and choose the required dataset (ISIC 2018 and Messidor) and then change the degree of heterogenity (eta) as required. you will get the desired distribution for each client.

## Model Structure
To choose the pretrained model, run model.py.

## Run FedMRL

After done with above process, you can run the FedMRL, our proposed method.

# Toxic comment classification
Repo was done as a test for deep nlp using the [toxic comment classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)) data from kaggle.

Another main motivation was to test out deep NLP models those used were:
- BERT
- ULMFiT
- PooledRNN

NOTE: Check output for results, contains fastai classification and pooled rnn results (both output sigmoid ouput (each class has percentage))

# Test BERT
Download model [here](https://drive.google.com/open?id=1bRiOF_CkyHRDZXFW1apf38Yobku5iDvn)
```sh
cd bert
python bert_test.py --text You are dumb # For single predict

or 

python bert_test.py --interactive # For console input
```

# BERT Model Training
Trained 3 times with 2 epochs each

## First cycle
![file_structure](https://github.com/edwin-19/Toxic-Comment-Classification/blob/master/assets/train-1.png?raw=true)  

## Second cycle
![file_structure](https://github.com/edwin-19/Toxic-Comment-Classification/blob/master/assets/train-2.png?raw=true)  

## Third cycle
![file_structure](https://github.com/edwin-19/Toxic-Comment-Classification/blob/master/assets/train-3.png?raw=true)  

# Enviroment
- Ubuntu 18.04
- Cuda 9
- Nvidia GTX 1080
- Cudnn 7.4

# Dependencies
- nltk
- tensorflow-gpu=1.9
- keras=2.2.4
- pytorch=1.1.0
- fastai
- torchvision=0.3.0

# ToDO
- [x] Train BERT model and test output
- [ ] Train FASTAi ULMFiT and test output
- [ ] Move from pytorch-bert-pretrained model package to transformers packege(latest)
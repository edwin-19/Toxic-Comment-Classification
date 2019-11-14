# Toxic comment classification
Repo was done as a test for deep nlp using the [toxic comment classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)) data from kaggle.

Another main motivation was to test out deep NLP models those used were:
- BERT - [paper](https://arxiv.org/abs/1810.04805) , [library](https://github.com/maknotavailable/pytorch-pretrained-BERT)
- ULMFiT
- PooledRNN

NOTE: Check output for results, contains fastai classification and pooled rnn results (both output sigmoid ouput (each class has percentage))

# Install 
```sh
git clone 

pip install -r requirements.txt
```
Download the toxic comment classification dataset from kaggle

Put in the folder
data/toxic_comment

# BERT
Make sure to put the fine tuned model inside the model folder within the bert folder

NOTE - for bert training check the notebook out
```sh
cd bert
python bert_test.py --text You are dumb # For single predict

or 

python bert_test.py --interactive # For console input
```

# Pooled RNN
```sh
python train_attention.py # train

python eval.py # Eval or generate csv output
```

# Models
| Model  | Download Link|
| ------------- | ------------- |
| BERT  | [Link](https://drive.google.com/open?id=1i5946rQuB8RWnZmHy6_advdZ6JoyMors)  |
| Pooled RNN  | [Link](https://drive.google.com/open?id=1UGcwDIXxwzIO9RHlDdy8L_Xt0tEeg1Fo) |

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

# Acknowledgement
- BERT and Fast AI code was heavily inspired by this repo check out the implementation [here](https://www.kaggle.com/abhikjha/jigsaw-toxicity-bert-with-fastai-and-fastai/notebook)

- Pooled RNN Keras code was also heavily inspired by the following repo check it out [here](https://github.com/zake7749/DeepToxic)

- For EDA the folowing github repo served as a backbone for the project those interested check it out [here](https://github.com/anmolchawla/Kaggle-Toxic-Comment-Classification-Challenge)
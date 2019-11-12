# Toxic comment classification
Repo was done as a test for deep nlp using the [toxic comment classification]((https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)) data from kaggle.

Another main motivation was to test out deep NLP models those used were:
- BERT
- ULMFiT
- PooledRNN

NOTE: Check output for results, contains fastai classification and pooled rnn results (both output sigmoid ouput (each class has percentage))

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
- [ ] Train BERT model and test output
- [ ] Train FASTAi ULMFiT and test output
- [ ] Move from pytorch-bert-pretrained model package to transformers packege(latest)
from fastai import *
from fastai.text import *
import bert_model
import numpy as np 

import argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--interactive", action='store_true'
    )
    
    argparse.add_argument(
        "--text", default="You are shit"
    )
    args = argparse.parse_args()
    bert = bert_model.load_bert_model('.')
    bert.load('head-3')
    
    if args.interactive:
        my_input = input("Enter command here, Enter 0 to stop\n")
        while my_input != "0":
            result = bert.predict(my_input)
            print('Test result:' + str(result[0])) 
            
            my_input = input("Enter command here, Enter 0 to stop\n")
    else:
        result = bert.predict(args.text)
        print('Test result:' + str(result[0])) 
    # bert.predict()
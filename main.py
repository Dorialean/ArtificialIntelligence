from sklearn.datasets import make_regression, make_classification, make_blobs
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #Наивный Байесовскй классификатор
    df = pd.read_table('datasets/SMSSpamCollection',
                       sep='/t',
                       header=None,
                       names=['label', 'message'])
    df['label'] = df.label.map({'not_spam':0,'spam':1})




if __name__ == '__main__':
    main()

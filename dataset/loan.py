import pandas as pd
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# df = pd.read_csv("./datasets/loan/bank_loan.csv")
# df_train = df.sample(frac=0.8)
# df_test = df.drop(df_train.index)
# df_train.to_csv("./datasets/loan/bank_loan_train.csv", index=False)
# df_test.to_csv("./datasets/loan/bank_loan_test.csv", index=False)

class Loan(Dataset_core):
    column_names = ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Personal Loan', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']

    target_name = 'Personal Loan'
    distance_column = ['Age', 'Experience', 'Income', 'CCAvg', 'Education', 'Mortgage', ]
    category_column = ['ID', 'ZIP Code', 'Family', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    model = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=400)
    task_type = "binclass"
    num_classes = 2
    tsne_drop_column = "ID"
    def __init__(self) -> None:
        pass
    def get_train_frame(self)-> pd.DataFrame:
        """_summary_
        Obtain the training set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the training set dataframe
        """
        df = pd.read_csv("./datasets/loan/bank_loan_train.csv")
        df['CCAvg'] = df['CCAvg'].str.replace('/', '.').astype('float64')
        return df


    def get_test_frame(self)->pd.DataFrame:
        """_summary_
        Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        df = pd.read_csv("./datasets/loan/bank_loan_test.csv")
        df['CCAvg'] = df['CCAvg'].str.replace('/', '.').astype('float64')
        return df
    
    

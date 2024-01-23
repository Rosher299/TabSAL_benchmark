from scipy.io import arff
import pandas as pd 
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestClassifier


# file_name='./datasets/diabetes/dataset_37_diabetes.arff'
# data,meta=arff.loadarff(file_name)
# df=pd.DataFrame(data)
# df.to_csv('./datasets/diabetes/diabetes.csv',index=False)
# df_train=df.sample(frac=0.8)
# df_test=df.drop(df_train.index)
# df_train.to_csv('./datasets/diabetes/diabetes_train.csv',index=False)
# df_test.to_csv('./datasets/diabetes/diabetes_test.csv',index=False)

class Diabetes(Dataset_core):
    column_names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
    target_name = 'class'
    distance_column = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age',]
    category_column = ['class']
    task_type = "binclass"
    num_classes = 2
    model = RandomForestClassifier(n_estimators=85, max_depth=12)



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
        data = pd.read_csv('./datasets/diabetes/diabetes_train.csv')
        return data


    def get_test_frame(self,)->pd.DataFrame:
        """_summary_
        Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        data = pd.read_csv('./datasets/diabetes/diabetes_test.csv')

        return data

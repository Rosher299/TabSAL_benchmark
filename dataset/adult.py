import pandas as pd
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Adult(Dataset_core):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
    target_name = 'income'
    distance_column = ['fnlwgt','age',  'educational-num','capital-gain', 'capital-loss', 'hours-per-week']
    category_column = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']

    key_fileds = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    sensitive_fields =['income']
    task_type = "binclass"
    num_classes = 2
    tsne_drop_column = "fnlwgt"
    # model = XGBClassifier(
    #     learning_rate= 0.1, 
    #     max_depth=7, 
    #     min_child_weight=3,
    #     gamma=0.1,  
    #     seed=477,
    #     n_estimators=1000,
    #     subsample= 0.8,
    #     colsample_bytree=0.8,
    #     objective= 'binary:logistic',
    #     )
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
        data = pd.read_csv("./datasets/adult/adult.data", names=self.column_names)
        return data
    
    def get_split_train_frame(self, frac:float)-> pd.DataFrame:
        """_summary_
        Obtain dataset slices proportionally
        Parameters
        ----------
        frac : float
            _description_ The proportion of training sets obtained from the dataset

        Returns
        -------
        pd.DataFrame
            _description_ the training set dataframe
        """
        data = pd.read_csv("./datasets/adult/adult.data", self.column_names)
        data = data.sample(frac=frac)
        # Reset Index
        data = data.reset_index(drop=True)
        return data


    def get_test_frame(self,)->pd.DataFrame:
        """_summary_
         Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        data = pd.read_csv("./datasets/adult/adult.test", names=self.column_names, skiprows=1)
        # Clear the last row of data in the income column Number
        data['income'] = data['income'].apply(lambda x: x[:-1])
        return data

    def format(self, data)->pd.DataFrame:
        """_summary_
        Convert the data type of the corresponding column in the dataset to Int type and adjust the column order of the data
        Parameters
        ----------
        data : _type_
            _description_ Unformatted data

        Returns
        -------
        pd.DataFrame
            _description_ Formatted data
        """
        data = data.copy()
        # Adjust column order
        data = data[Adult.column_names]
        # Traverse the data types of each column and convert the float type to an int type
        for col in data.columns:
            # If it is a float type
            if data[col].dtype == 'float64':
                # Convert it to int type
                data[col] = data[col].astype(int)
        
        # Delete any values as? Rows of
        # for col in data.columns:
        #     data = data[data[col] != " ?"]
        # Reset index
        # data = data.reset_index(drop=True)
        return data
    

    

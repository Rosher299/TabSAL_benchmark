import pandas as pd
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# def clear_df(df:pd.DataFrame)->pd.DataFrame:
#     """_summary_
#     Clean up the dataset
#     Parameters
#     ----------
#     df : pd.DataFrame
#         _description_

#     Returns
#     -------
#     pd.DataFrame
#         _description_
#     """
#     df = df.copy()
#     # Convert time to timestamp
#     df['issue_date'] = pd.to_datetime(df['issue_date']).astype(int) / 10**9
#     df['listing_date'] = pd.to_datetime(df['listing_date']).astype(int) / 10**9
#     df['pet_category'] = df['pet_category'].replace(4, 3)
#     df['condition'] = df['condition'].fillna(4)
#     return df
# df = pd.read_csv("./datasets/buddy/buddy.csv")
# df = clear_df(df)
# # Divide the training and testing sets into 4:1 parts
# df_train = df.sample(frac=0.8)
# df_test = df.drop(df_train.index)
# # Save Dataset
# df_train.to_csv("./datasets/buddy/buddy_train.csv", index=False)
# df_test.to_csv("./datasets/buddy/buddy_test.csv", index=False)

class Buddy(Dataset_core):
    column_names = ['pet_id', 'issue_date', 'listing_date', 'condition', 'color_type',
       'length(m)', 'height(cm)', 'X1', 'X2', 'breed_category',
       'pet_category']
    target_name = 'pet_category'
    distance_column = ['issue_date', 'listing_date', 'length(m)', 'height(cm)', 'X1', 'X2']
    category_column = ['pet_id',  'condition', 'color_type', 'breed_category', 'pet_category']
    model = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=400)
    task_type = "multiclass"
    num_classes = 4
    tsne_drop_column = "pet_id"
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
        df = pd.read_csv("./datasets/buddy/buddy_train.csv")
        return df


    def get_test_frame(self)->pd.DataFrame:
        """_summary_
        Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        df = pd.read_csv("./datasets/buddy/buddy_test.csv")
        return df
    
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
        data = self.get_train_frame()
        data = data.sample(frac=frac)
        # Reset Index
        data = data.reset_index(drop=True)
        return data
    
    def mle_evluation(self, df_train, model=None) -> dict:
        """_summary_
        mle testing
        Parameters
        ----------
        df_train : _type_ Composite dataset
            _description_

        Returns acc, f1, auc
        -------
        _type_
            _description_ Evaluating indicator
        """
        model = self.model if model == None else model
        df_test = self.get_test_frame()
        # training model
        df_train = self.digitization(df_train)
        if len(set(df_train['pet_category'])) == 2:
            # Add a row so that there are 4 categories in the training set
            df_train = df_train.append(df_train.iloc[0])
            df_train.iloc[-1, -1] = 0
            df_train = df_train.append(df_train.iloc[0])
            df_train.iloc[-1, -1] = 3

        df_test = self.digitization(df_test)
        model.fit(df_train.drop(self.target_name, axis=1), df_train[self.target_name])    
        # Calculate accuracy, F1 score, and AUC
        x = df_test.drop(self.target_name, axis=1)
        y = df_test[self.target_name]
        y_ = model.predict(x)
        acc = accuracy_score(y, y_)
        f1 = f1_score(y, y_, average='macro')
        y_prob = model.predict_proba(x)
        auc = roc_auc_score(y, y_prob, multi_class='ovo')
        return {'acc':acc, 'f1':f1, 'auc':auc}
    
    def digitization(self, data:pd.DataFrame):
        """_summary_
        Digitization of dataset attributes
        Parameters
        ----------
        data : pd.DataFrame
            _description_ Dataset after attribute digitization
        """
        data  = data.copy()
        # Traverse the data types of each column in the data
        for col in data.columns:
            # If it is an object type
            if data[col].dtype == 'object':
                # Convert it to category type
                data[col] = pd.Categorical(data[col]).codes
        return data
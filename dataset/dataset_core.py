import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sdmetrics.single_table import CategoricalCAP
from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
class Dataset_core:
    column_names:list = []
    target_name:str = ""
    # The name of the distance column
    distance_column:list = []
    # The name of the category column
    category_column:list = []
    # The name of the primary key column
    key_fileds:list = []
    # Name of sensitive column
    sensitive_fields:list = []

    task_type:str = ""
    num_classes:int = 0
    model = RandomForestClassifier(n_estimators=85, max_depth=12)
    model_xgb = XGBClassifier(
        learning_rate= 0.1, 
        max_depth=7, 
        min_child_weight=3,
        gamma=0.1,  
        n_estimators=200,
        subsample= 0.8,
        colsample_bytree=0.8,
        )
    model_lr = LogisticRegression(solver="sag", max_iter=10000)
    model_dt = DecisionTreeClassifier()
    tsne_drop_column = ""
    def __init__(self) -> None:
        pass
    def get_train_frame(self)-> pd.DataFrame:
        raise NotImplementedError("This has to be overwritten but the subclasses")

    def get_split_train_frame(self,frac:float)-> pd.DataFrame:
        raise NotImplementedError("This has to be overwritten but the subclasses")
    def get_test_frame(self)->pd.DataFrame:
        raise NotImplementedError("This has to be overwritten but the subclasses")

    def evaluate(self, df_) -> dict:
        df = self.get_test_frame()
        y = df[self.target_name]
        y_ = df_[self.target_name]
        if self.task_type == "regression":
            mape = mean_absolute_percentage_error(y, y_)
            mse = mean_squared_error(y, y_)
            return {"mape":mape, "mse":mse}
        else:
            if y.dtype == 'object':
                # Convert it to category type
                # Identify unique categories
                unique_categories = set(y) | set(y_)

                # Create a common set of categories
                common_categories = pd.Categorical(list(unique_categories))
                # Convert to categorical codes using the common set
                y = pd.Categorical(y, categories=common_categories).codes
                y_ = pd.Categorical(y_, categories=common_categories).codes
            acc = accuracy_score(y, y_)
            f1 = f1_score(y, y_, average='weighted')
            return {'acc':acc, 'f1':f1}
        
    def tsne_plot(self, df: pd.DataFrame):
        """_summary_
        Draw a tsne graph of the dataset
        Parameters
        ----------
        df : pd.DataFrame
            _description_
        """
        df = self.digitization(df)
        if self.tsne_drop_column != "":
            df = df.drop(self.tsne_drop_column, axis=1)
        x = df.drop(self.target_name, axis=1)
        y = df[self.target_name]
        tsne = TSNE(n_components=2, random_state=0, perplexity=50)
        x_tsne = tsne.fit_transform(x)
        plt.figure(figsize=(12, 8))
        # Draw a scatter plot with different colors and styles for different labels
        for label in set(y):
            plt.scatter(x_tsne[y == label, 0], x_tsne[y == label, 1], label=label, s=2)
        plt.show()
        # Return image
        return plt
        

    def format(self, data)->pd.DataFrame:
        """_summary_
        Convert the data type of the corresponding column in the dataset to Int type and adjust the column order of the data
        Parameters
        ----------
        data : _type_
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        data = data.copy()
        # Adjust column order
        data = data[self.column_names]
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
            _description_
        """
        model = self.model if model == None else model
        df_test = self.get_test_frame()
        # # training model
        df_train = self.digitization(df_train)
        df_test = self.digitization(df_test)
        model.fit(df_train.drop(self.target_name, axis=1), df_train[self.target_name])    
        # Calculate accuracy, F1 score, and AUC
        x = df_test.drop(self.target_name, axis=1)
        y = df_test[self.target_name]
        y_ = model.predict(x)
        acc = accuracy_score(y, y_)
        f1 = f1_score(y, y_, average='weighted')
        
        # This AUC calculation method is only applicable to binary classification
        y_prob=model.predict_proba(x)[:,1]
        auc = roc_auc_score(y, y_prob)
        return {'acc':acc, 'f1':f1, 'auc':auc}
    
    def cap_evluation(self, df_gen):
        return CategoricalCAP.compute(
        real_data=self.get_train_frame(),
        synthetic_data=df_gen,
        key_fields=self.key_fileds,
        sensitive_fields=self.sensitive_fields,
    )

    # https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement
    def similarity_evaluation(self, df_gen) ->dict:
        """_summary_
        Calculate similarity based on a bar chart
        Parameters
        ----------
        df_gen : _type_
            _description_

        Returns ks, tv
        -------
        ks is the similarity of continuous columns, and tv is the similarity of discrete columns
        """
        ks=0
        for col in self.distance_column:
            ks += KSComplement.compute(
                real_data=self.get_train_frame()[col],
                synthetic_data=df_gen[col],
            )
        ks = ks/len(self.distance_column) if len(self.distance_column) > 0 else 0
        tv = 0
        for col in self.category_column:
            tv += TVComplement.compute(
                real_data=self.get_train_frame()[col],
                synthetic_data=df_gen[col],
            )
        tv = tv/len(self.category_column) if len(self.category_column) > 0 else 0

        ks_target = 0
        tv_target = 0
        if self.target_name in self.distance_column:
            ks_target = KSComplement.compute(
                real_data=self.get_train_frame()[self.target_name],
                synthetic_data=df_gen[self.target_name],
            )
        elif self.target_name in self.category_column:
            tv_target = TVComplement.compute(
                real_data=self.get_train_frame()[self.target_name],
                synthetic_data=df_gen[self.target_name],
            )

        result = {'ks':ks, 'tv':tv, 'ks_target':ks_target, 'tv_target':tv_target}
        return result
    
    def target_similarity_evaluation(self, df_gen) -> dict:
        if self.target_name in self.distance_column:
            ks = KSComplement.compute(
                real_data=self.get_train_frame()[self.target_name],
                synthetic_data=df_gen[self.target_name],
            )
            return {'ks':ks}
        elif self.target_name in self.category_column:
            tv = TVComplement.compute(
                real_data=self.get_train_frame()[self.target_name],
                synthetic_data=df_gen[self.target_name],
            )
            return {'tv':tv}
    
    def distance_evaluation(self, df_gen):
        df_train = self.get_train_frame()

        def distance(item1, item2, cat_or_dis):
            if cat_or_dis == "cat":
                dis = np.sum(item1 != item2)
            else:
                dis = np.sum(np.abs(item1 - item2))

            return dis
        dcr = []
        nndr = []

        df_train_cat_values = df_train[self.category_column].values
        df_train_dis_values = df_train[self.distance_column].values

        for (row_cat, row_dis) in tqdm(zip(df_gen[self.category_column].values, df_gen[self.distance_column].values), total=df_gen.shape[0]):
            cat_distances = np.apply_along_axis(lambda x: distance(row_cat, x, "cat"), axis=1, arr=df_train_cat_values)
            dis_distances = np.apply_along_axis(lambda x: distance(row_dis, x, "dis"), axis=1, arr=df_train_dis_values)
            sorted_distances = np.sort(cat_distances + dis_distances)
            min_dis_1 = sorted_distances[0]
            min_dis_2 = sorted_distances[1]

            dcr.append(min_dis_1)
            nndr.append(min_dis_1 / min_dis_2 if min_dis_2 != 0 else 0)

        return dcr, nndr

    
    def digitization(self, data:pd.DataFrame):
        """_summary_
        Digitization of dataset attributes
        Parameters
        ----------
        data : pd.DataFrame
            _description_
        """
        data  = data.copy()
        # Traverse the data types of each column in the data
        for col in data.columns:
            # If it is an object type
            if data[col].dtype == 'object':
                # Convert it to category type
                data[col] = pd.Categorical(data[col]).codes
        return data
    
    def dedigitization(self, data)->pd.DataFrame:
        """_summary_
        Dedigitize label columns
        Parameters
        ----------
        data : _type_
            _description_
        Digitized label columns
        Returns
        -------
        pd.DataFrame
            _description_
        """
        df_origin = self.get_train_frame()
        categories = pd.Categorical(df_origin[self.target_name]).categories
        data = pd.Categorical.from_codes(data, categories=categories).tolist()
        return data
    
    
    


        
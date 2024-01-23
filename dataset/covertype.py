from sklearn.datasets import fetch_covtype
import pandas as pd
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


# df = fetch_covtype(as_frame=True).frame 
# # Divide the training and testing sets into 4:1 parts
# df_train = df.sample(frac=0.8)
# df_test = df.drop(df_train.index)
# # Save Dataset
# df_train.to_csv("./datasets/covertype/covertype_train.csv", index=False)
# df_test.to_csv("./datasets/covertype/covertype_test.csv", index=False)

class Covertype(Dataset_core):
    column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0',
       'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3',
       'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3',
       'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7',
       'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11',
       'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15',
       'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19',
       'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23',
       'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27',
       'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31',
       'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35',
       'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39',
       'Cover_Type']

    target_name = 'Cover_Type'
    distance_column = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points',]
    category_column = ['Wilderness_Area_0',
       'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3',
       'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3',
       'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7',
       'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11',
       'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15',
       'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19',
       'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23',
       'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27',
       'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31',
       'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35',
       'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39',
       'Cover_Type']
    model = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=400)
    task_type = "multiclass"
    num_classes = 7
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
        df = pd.read_csv("./datasets/covertype/covertype_train.csv")
        return df


    def get_test_frame(self)->pd.DataFrame:
        """_summary_
        Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        df = pd.read_csv("./datasets/covertype/covertype_test.csv")
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
    
    def mle_evluation(self, df_train) -> dict:
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
        model = self.model
        df_test = self.get_test_frame()
        # training model
        df_train = self.digitization(df_train)
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
            if data[col].dtype == 'object' or col == self.target_name: # Since this label is marked starting from 1, it also needs to be converted
                # Convert it to category type
                data[col] = pd.Categorical(data[col]).codes
        return data
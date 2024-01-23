import pandas as pd
from dataset.dataset_core import Dataset_core
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


# df = pd.read_csv("./datasets/insurance/insurance.csv")
# df_train = df.sample(frac=0.8)
# df_test = df.drop(df_train.index)
# df_train.to_csv("./datasets/insurance/insurance_train.csv", index=False)
# df_test.to_csv("./datasets/insurance/insurance_test.csv", index=False)

class Insurance(Dataset_core):
    column_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    target_name = 'charges'
    distance_column = ['age',  'bmi', 'charges']
    category_column = ['sex', 'children', 'smoker', 'region']
    model = RandomForestRegressor( n_estimators=1000)
    task_type = "regression"
    num_classes = 0
    model_xgb = XGBRegressor()
    model_dt = DecisionTreeRegressor()
    model_lr = LinearRegression()
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
        df = pd.read_csv("./datasets/insurance/insurance_train.csv")
        return df


    def get_test_frame(self)->pd.DataFrame:
        """_summary_
        Obtain the test set dataframe
        Returns
        -------
        pd.DataFrame
            _description_ the test set dataframe
        """
        df = pd.read_csv("./datasets/insurance/insurance_test.csv")
        return df
    
    def mle_evluation(self, df_train, model=None) -> dict:
        """_summary_
        mle testing
        Parameters
        ----------
        df_train : _type_ Composite dataset
            _description_

        Returns mape,mse
        -------
        _type_
            _description_ Evaluating indicator
        """
        model = self.model if model == None else model
        df_test = self.get_test_frame()
        # training model
        df_train = self.digitization(df_train)
        df_test = self.digitization(df_test)
        model.fit(df_train.drop(self.target_name, axis=1), df_train[self.target_name])    
        # mape,mse
        x = df_test.drop(self.target_name, axis=1)
        y = df_test[self.target_name]
        y_ = model.predict(x)
        mape = mean_absolute_percentage_error(y, y_)
        mse = mean_squared_error(y, y_)
        return {"mape":mape, "mse":mse}

    def dedigitization(self, data):
        return data 
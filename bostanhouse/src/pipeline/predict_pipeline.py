import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        crim: float,
        zn: float,
        indus: float,
        chas: int,
        nox: float,
        rm: float,
        age: float,
        dis: float,
        rad: int,
        tax: float,
        ptratio: float,
        b: float,
        lstat: float):

        self.crim = crim
        self.zn = zn
        self.indus = indus
        self.chas = chas
        self.nox = nox
        self.rm = rm
        self.age = age
        self.dis = dis
        self.rad = rad
        self.tax = tax
        self.ptratio = ptratio
        self.b = b
        self.lstat = lstat

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "crim": [self.crim],
                "zn": [self.zn],
                "indus": [self.indus],
                "chas": [self.chas],
                "nox": [self.nox],
                "rm": [self.rm],
                "age": [self.age],
                "dis": [self.dis],
                "rad": [self.rad],
                "tax": [self.tax],
                "ptratio": [self.ptratio],
                "b": [self.b],
                "lstat": [self.lstat],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import pandas as pd
import numpy as np    
import time
# from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
import re
import time

def test_clean():
    print("test clean")

class ModuleError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, debug = True):
        self._start_time = None
        self.debug = debug

    def start(self,message='Elapsed time'):
        """Start a new timer"""
        if self._start_time != None:
            raise ModuleError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        self.message = message

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time == None:
            raise ModuleError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{self.message}: {elapsed_time:0.4f} seconds")
    
class DataModule():
    def __init__(self, date_col='timestamp',
                 # imputed_val = -9,
                 scale_method='z', 
                 dr_algorithm='pca', dr_components=2,
                #  date_features=['minute','hour','dow'],
                 # date_features=['h_m_grp'],
                 seed = 55,
                 round = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self._fit = None
        self.date_col = date_col
        # self.date_features = date_features
        self.scale_method = scale_method
        self.dr_algorithm = dr_algorithm
        self.dr_components = dr_components
        # self.imputed_val = imputed_val
        self.imputed_val = None
        self.round = round
        self.seed = seed
        self.scaler = None
        self.dr = None

    ##----------<Start: th== code block can be changed, based on a model>----------##
    def _scale_data(self, data, method='train'):
        if self.scale_method != None:
            if method == 'train':
                data = self.scaler.fit_transform(data.loc[:,self.column].values.reshape(-1,self.n_features))
            elif method == 'inference':
                data = self.scaler.transform(data.loc[:,self.column].values.reshape(-1,self.n_features))
        elif self.scale_method == None:
            data = data.values
        return data

    def _dimensionality_reduction(self, data, method='train'):
        if self.dr_algorithm != None:
            if method == 'train':
                data = self.dr.fit_transform(data)
            elif method == 'inference':
                data = self.dr.transform(data)
        elif self.dr_algorithm == None:
            data = data
        return data
    ##----------<End: th== code block can be changed, based on a model>----------##

    def _init_module(self, data, column, method):
        if method == 'train':
            self.column = column
            self.n_features = 1 if isinstance(column, str) else len(column)
            self.features = [self.date_col, self.column] if isinstance(column, str) else [self.date_col] + list(self.column)
            self.imputed_val = data[self.column].min() - 10
            self._fit = True
            # self.features = [self.column] + self.date_features

            if self.scale_method == 'z':
                self.scaler = StandardScaler()
            elif self.scale_method == 'power':
                self.scaler = PowerTransformer(method='yeo-johnson')

            if self.dr_algorithm == 'pca':
                self.dr = PCA(n_components=self.dr_components, random_state=self.seed)
            else:
                self.dr = PCA(n_components=self.dr_components, random_state=self.seed)
        # self.date_idx = data.loc[:,self.date_col].copy()
    def impute_value(self, data):
        data[self.column] = data[self.column].fillna(self.imputed_val)
        return data
    
    def _cleansing(self, data):
        data = self.impute_value(data)
        return data

    def _extract_datetime(self, data):
        data['day'] = data[self.date_col].dt.day
        data['dow'] = data[self.date_col].dt.dayofweek + 1 # 1=monday, 7=sunday
        data['day_name'] = data[self.date_col].dt.day_name()
        data['minute'] = data[self.date_col].dt.minute
        data['hour'] = data[self.date_col].dt.hour
        data['date'] = data[self.date_col].dt.date
        comb_date = lambda x: f"0{x}" if len(str(x))==1 else str(x)
        data['h_m'] = data['hour'].apply(comb_date) + '-' + data['minute'].apply(comb_date)
        data['hm_grp'] = data['h_m'].map({k:v for k,v in zip(data['h_m'].unique(),range(data['h_m'].nunique()))})
        data['d_h_m'] = data['dow'].apply(comb_date) + '-' + data['hour'].apply(comb_date) + '-' + data['minute'].apply(comb_date)
        data['dhm_grp'] = data['d_h_m'].map({k:v for k,v in zip(data['d_h_m'].unique(),range(data['d_h_m'].nunique()))})
        # data['d_h_m'] = data['hour'].apply(lambda x: f"0{x}" if len(str(x))==1 else str(x)) + '-' + data['minute'].apply(lambda x: f"0{x}" if len(str(x))==1 else str(x))
        # data['h_m_grp'] = data['h_m'].map({k:v for k,v in zip(data['h_m'].unique(),range(288))})
        # self.features = data.columns.copy()
        # return data.loc[:,self.features]
        return data

    ##----------<Start: th== code block can be changed, based on a model>----------##
    def model_methodology(self, data, method):
        return data.round(4)
    ##----------<End: th== code block can be changed, based on a model>----------##

    def _preprocess(self, X, column, method):
        self._init_module(X, column, method)
        data = X.loc[:,self.features].copy()
        data = self._cleansing(data)
        data = self._extract_datetime(data)
        data[self.column] = self._scale_data(data, method).round(4)
        # data = self._dimensionality_reduction(data, method).round(4)
        # data = self.model_methodology(data, method)
        return data
        
    def build_train(self, X, column):
        return self._preprocess(X, column, method='train')

    def build_inference(self, X):
        if self._fit == None:
            raise ModuleError(f"apply GulfDataModule.build_train(X, column) before running GulfDataModule.build_inference()")
        return self._preprocess(X, column=self.column, method='inference')
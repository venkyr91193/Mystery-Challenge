import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.externals import joblib


class Generate:
  def __init__(self, args):
    self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'model')
    # loading the model
    self.model = joblib.load(os.path.join(self.model_dir,'saved_model.pkl'))
    # reading the files
    self.filepath = os.path.join(args.input_path,'gwlq.csv')
    self.df = pd.read_csv(self.filepath)
    # catalog file
    self.filepath_catalog = os.path.join(args.input_path,'catalog.csv')
    self.catalog_df = pd.read_csv(self.filepath_catalog)
    self.temp_df = self.df[['x','y','z']]

  def is_sub(self, list_1, list_2)->bool:
    '''
    To find if a list is a subset of another list
    '''
    ln = len(list_1)
    for idx in range(len(list_2) - ln + 1):
      if all(list_1[idy] == list_2[idx+idy] for idy in range(ln)):
        return True
    return False

  def generate(self):
    '''
    Generate hidden image
    '''
    X_data = self.df.iloc[:,3:]
    Y_data = self.model.predict(X_data)
    print()
    array = list()
    grouped_df = self.df.groupby(['z'])
    for key,value in grouped_df:
      X_data = value.iloc[:,3:]
      Y_data = self.model.predict(X_data)
      array.append(np.count_nonzero(Y_data))
    print()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path",
                        type=str,
                        required=False,
                        default=r'C:\Users\venks\Downloads\Others\datasets',
                        help="Data source path for the csv files")
  args = parser.parse_args()
  temp_obj = Generate(args)
  temp_obj.generate()

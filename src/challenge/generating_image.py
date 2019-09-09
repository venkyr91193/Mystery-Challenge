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
    self.temp_df = self.df[['x','y','z']]

  def generate(self):
    '''
    Generate hidden image
    '''
    X_data = self.df.iloc[:,3:]
    Y_data = self.model.predict(X_data)
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

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d


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
    self.abcd = os.path.join(args.input_path,'tests','abcdefghijklmnopqrstuvwxyz_0.csv')
    self.test_0 = os.path.join(args.input_path,'tests','this-is-a-test_0.csv')
    self.test_5 = os.path.join(args.input_path,'tests','this-is-a-test_5.csv')
    self.test_10 = os.path.join(args.input_path,'tests','this-is-a-test_10.csv')
    self.test_20 = os.path.join(args.input_path,'tests','this-is-a-test_20.csv')
    self.catalog_df = pd.read_csv(self.filepath_catalog)
    self.df_abcd = pd.read_csv(self.abcd)
    self.df_test_0 = pd.read_csv(self.test_0)
    self.df_test_5 = pd.read_csv(self.test_5)
    self.df_test_10 = pd.read_csv(self.test_10)
    self.df_test_20 = pd.read_csv(self.test_20)

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
    for df in [self.df,self.df_abcd,self.df_test_0,self.df_test_5,self.df_test_10,self.df_test_20]:
      temp_df = df[['x','y','z']]
      X_data = temp_df.iloc[:,3:]
      Y_data = self.model.predict(X_data)
      Y_data = np.logical_not(np.asarray(Y_data))
      temp_df['spiral'] = Y_data
      temp = temp_df[(temp_df['spiral'] == True)]
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      xdata = temp['x'].values
      ydata = temp['y'].values
      zdata = temp['z'].values
      ax.scatter3D(xdata, ydata, zdata)
      plt.show()
    print()

    # grouping and checking the number of true spirals per disc of the file gwlq
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
                        default=r'C:\Users\Venkataramana R\Downloads\Others\datasets',
                        help="Data source path for the csv files")
  args = parser.parse_args()
  temp_obj = Generate(args)
  temp_obj.generate()

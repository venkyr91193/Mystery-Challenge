import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

def analyse(args):
  # reading the files
  filepath_1 = os.path.join(args.input_path,'gwlq.csv')
  filepath_2 = os.path.join(args.input_path,'catalog.csv')
  df_1 = pd.read_csv(filepath_1)
  df_2 = pd.read_csv(filepath_2)

  # plotting the x,y,z of the data
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  xdata = df_1['x'].values
  ydata = df_1['y'].values
  zdata = df_1['z'].values
  ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
  plt.show()
  print()

  # single plane projection leads to a disk in the form of circle
  df_temp = df_1[df_1['z'] == 0]
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  xdata = df_temp['x'].values
  ydata = df_temp['y'].values
  zdata = df_temp['z'].values
  ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
  plt.show()
  print()

  # plotting the row data into a 2d array based on the columns
  # just for the first row
  temp = np.ndarray((50,50))
  for num in df_1.columns[3:]:
    i = int(num.split(',')[0])
    j = int(num.split(',')[1])
    temp[i][j] = df_1.iloc[0][num]

  # seeing a spiral image which is very helpful for the next clue.
  plt.imshow(temp)
  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path",
                        type=str,
                        required=False,
                        default=r'C:\Users\venks\Downloads\Others\datasets',
                        help="Data source path for the csv files")
  args = parser.parse_args()
  analyse(args)

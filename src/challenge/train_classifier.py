import argparse
import os
import pickle

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Train:
  def __init__(self, args):
    self.filepath = os.path.join(args.input_path,'catalog.csv')
    self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'model')
    self.df = pd.read_csv(self.filepath)
    # initializaing the model objects
    self.LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    self.SVM = svm.LinearSVC()
    self.RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    self.NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

  def train(self):
    '''
    Function to train the logistic regression
    '''
    X_data = self.df.iloc[:,1:]
    Y_data = self.df.iloc[:,0]
    x_train = X_data.iloc[:int(0.8*len(X_data)),:]
    x_test = X_data.iloc[int(0.8*len(X_data)):,:]
    y_train = Y_data.iloc[:int(0.8*len(Y_data))]
    y_test = Y_data.iloc[int(0.8*len(Y_data)):]
    # using 80, 20 split for train and validation
    self.LR.fit(x_train,y_train)
    self.SVM.fit(x_train,y_train)
    self.RF.fit(x_train,y_train)
    self.NN.fit(x_train,y_train)
    print('The accuracy of the Logistic Regression model over the test datset is:',self.LR.score(x_test,y_test))
    print('The accuracy of the Support Vector Machines model over the test datset is:',self.SVM.score(x_test,y_test))
    print('The accuracy of the Random Forest model over the test datset is:',self.RF.score(x_test,y_test))
    print('The accuracy of the Multi Layer Perceptron model over the test datset is:',self.NN.score(x_test,y_test))
  
  def save_model(self):
    '''
    Function to save the model
    '''
    '''
    The accuracy of the Logistic Regression model over the test datset is: 0.994
    The accuracy of the Support Vector Machines model over the test datset is: 0.993
    The accuracy of the Random Forest model over the test datset is: 0.936
    The accuracy of the Multi Layer Perceptron model over the test datset is: 0.663
    '''
    # saving the best model. Logistic Regression
    # Output a pickle file for the model
    joblib.dump(self.LR, os.path.join(self.output_dir,'saved_model.pkl'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path",
                        type=str,
                        required=False,
                        default=r'C:\Users\venks\Downloads\Others\datasets',
                        help="Data source path for the csv files")
  args = parser.parse_args()
  temp_obj = Train(args)
  temp_obj.train()
  temp_obj.save_model()

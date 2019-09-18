# UNDERSTANDING THE PROBLEM

There are 7 datasets in the folders.

6 out of 7 datasets have the same format with columns x,y and z and 2500 columns. The catalog file represents some data which marks the 2500 columns as spiral or not spiral.

# Initial analysis of the data. (analysing_dataset.py)

The aim of this python code is to analyse the dataset and see what it is made up of before proceeding further into the problem.
The python code does the plotting of the x,y and z columns of the dataset into a 3D scatter plot using the matplotlib library in python.
This analysis gave me an idea of how the x,y,z formed together. The result was a cylinder with a height of 24 unites and a radius of 1 unit with x and y ranging from (-1,1). Refer [images\xyz.png]

The next was to see how many points were in a single disc of a cylinder. There are 12500 points for each disc whose number varies from 0 to 23 (total 24). Refer [images\single_plane_disc.png]

The next step was to analyse the 2500 columns that were present in each row of the dataset. The columns were named 0,0 which gave me a clue that it might be representing the indexes of a matrix. So the columns were ranging from 0,0 to 0,49 and from 49,0 to 49,49. So if we group it into the required indexes, then it gives us 50*50 matrix.

After getting the 50*50 matrix, I plotted the matrix in a simple manner which gave me the next clue. The image represented a spiral shape for the first row. If you keep checking for the subsequent rows, the images may or may not represent the spiral. Refer [images\spiral.png]

# catalog dataset (train_classifier.py)

The catalog dataset consists of the same 2500 columns which when plotted may give a spiral or non spiral with the corresponding labels in the column called spiral.
So the next step was to easily classify each row of the dataset into spiral or non spiral just by using the data from the row. This can be done by using a simple logistic regression of the series data of each row. So the modelling of the series is done in the python code train_classifier.py. The fifferent models along with different accuracy is explained in the code. The model with the highest accuracy, which is the logistic regression was choosen and its stored in the mdoel folder.

# predicting the result of each row in the datasets (generate_image.py)

The model stored in the model folder is again loaded in the generate_image.py python file for classifying each row of a dataset into spiral or not spiral. 

Analysing the file inside [tests\abcdefghijklmnopqrstuvwxyz_0.csv] gave me an interesting plot which is inside [images\abcd.png] which when rotated and seen from the top view gave an intresting spiral like image. The columns of 'spiral' data which is predicted by the model is inverted using logical_not and plotted to see the above result.

Analysing the files this_is_a_test_0.csv to this_is_a_test_20.csv gave me similar results from the top view except the angle of view was shifed by 90 deg for each file.

The results obtained from the file gwlq.csv for each separate disks, say example for z = 0 to z = 23 closely resembeled the constellations. Althought it was totally not equal since there was lot of noise to be filtered out.  

# Assignment 6 - IN4110
#### By David Andreas Bordvik


### 6.1 - Handling the data:
The script reads in data from a csv file using pandas. The data is spilt into
traning set and validation set. 

#### Usage:
For help on how to run the script:
`python3 data.py -h`


### 6.2 - Fitting a machine learning model:
This script fits various sklearn classifiers based on features. The function 'fit' 
returns the trained classifier.

#### Usage:
For help on how to run the script:
`python3 fitting.py -h`


### 6.3 - Visualizing the classifier:
This script `visualize.py` contains a function that visualize the classifier when choosen feature subset
has only 2 features. The function returns a matplotlib figure.
 

### 6.4 - Visualization through a web app:
This `web_visualization.py` flask app uses the script from 6.3 to generate a plot of the classifier
and displays it on a web page.  


### 6.5 - Interactive visualization:
6.4 is modified make the user able to change classifier and features in the web page.


#-------------------------------------------------------------------------
# AUTHOR: Adrian Alcoreza
# FILENAME: decision_tree.py
# SPECIFICATION: Given data from contact_lens.csv, output a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: Roughly 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# Creating a dictionary that will map the original categorical feature values into numbers.
transformed_values = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3, 
                      "Myope": 1, "Hypermetrope": 2,
                      "Yes": 1, "No": 2,
                      "Normal": 1, "Reduced": 2}

# Getting each except the last column of the db and replacing their values with numerical ones.
X = [row[:-1] for row in db]
X = [[transformed_values[old_value] for old_value in row] for row in X]

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
# Getting the last column of the db and replacing its values with numerical ones.
Y = [row[-1:][0] for row in db]
Y = [1 if value == "Yes" else 2 for value in Y]

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
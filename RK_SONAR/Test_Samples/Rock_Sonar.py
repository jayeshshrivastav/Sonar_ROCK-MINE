# Importing all the required dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data using pandas
data = pd.read_csv(r"/home/aditya/Desktop/RK_SONAR/sonar_data.csv",header = None)

# Printing the first five columns
data.head()

# Number of rows and columns in the data
data.shape

# finding if any missing value available
data.isnull().sum()

# using info for the data 
data.info()

# using describe function finding more statistical measure about data
data.describe()

# finding no. of rocks and mines in the data
data[60].value_counts()

data.groupby(60).mean()


# seperating data and label
x = data.drop(columns = 60,axis = 1)
y = data[60]

print(x)

print(y)

# spliting training and test data
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)

print(x.shape,X_test.shape,X_train.shape)

# Model training --> Logistic Regression model
model = LogisticRegression()

# training the Logistic Regression model using training data
model.fit(X_train,Y_train)

# Model Evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy on training data: " ,training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("Accuracy on test data: " ,test_data_accuracy)

# Making a Predictive system
input_data = (0.0262,0.0582,0.1099,0.1083,0.0974,0.2280,0.2431,0.3771,0.5598,0.6194,0.6333,0.7060,0.5544,0.5320,0.6479,0.6931,0.6759,0.7551,0.8929,0.8619,0.7974,0.6737,0.4293,0.3648,0.5331,0.2413,0.5070,0.8533,0.6036,0.8514,0.8512,0.5045,0.1862,0.2709,0.4232,0.3043,0.6116,0.6756,0.5375,0.4719,0.4647,0.2587,0.2129,0.2222,0.2111,0.0176,0.1348,0.0744,0.0130,0.0106,0.0033,0.0232,0.0166,0.0095,0.0180,0.0244,0.0316,0.0164,0.0095,0.0078)

#changing the input as numpy array
input_data_as_numpy = np.array(input_data)

# reshape the np array as we are prediction for one instance
input_data_reshape = input_data_as_numpy.reshape(1,-1)

prediction = model.predict(input_data_reshape)

if prediction == "M":
    print("Its a Mine")
else:
    print("Its a rock")
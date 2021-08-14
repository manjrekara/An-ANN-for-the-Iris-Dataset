# Training an ANN on the Iris dataset
# by Atharva Manjrekar

# Import relevant libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn.preprocessing import normalize

# Locate and save dataset as df
df = pd.read_csv('iris_data.txt', header = None)
# Assign column labels to the dataframe
df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
# Save df as a separate .csv file to work on
df.to_csv('new_iris_data.csv', index = None)
# Visualize the df
print(df.sample(10))

# Differentiate features (x) and classes (y)
x = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Convert class category (y) as a vector where 0 = iris-setosa, 1 = iris-versicolor, 2 = iris-virginica
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
y = pd.get_dummies(y1).values

print("Shape of x: ",x.shape)
print("Shape of y: ",y.shape)
print("--------------------------------")
# Shows the first three rows of x and y matrix
print("Examples of x: \n",x[:3])
print("Examples of y: \n",y[:3])
print("--------------------------------")

# Normalized x values will be 0 to 1
x_normalized = normalize(x,axis=0)
print("Examples of a normalised x: \n",x_normalized[:3])
print("\n--------------------------------")
# x normalized is set equal to the training and testing values for x
x_train = x_normalized
x_test = x_normalized

# Split the 150 datapoints into 80% training (120) and 20% testing (30)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Visualize the dataframe as a correlation matrix (optional)
sns.set(style='ticks')
sns.color_palette("icefire")
sns.pairplot(df.iloc[:,0:5],hue="Species")

# Visualize the training and testing shapes x and y 
# Here we can confirm if the train/test split was appropriate to the percentage we chose
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)
print("\n--------------------------------")

# Create a sequential model that has 3 hidden layers and a dropout layer  
# 1 input layer (4 features of the flower) and 1 output layer (3 species)
model = Sequential()

# First hidden layer with 500 neurons
model.add(Dense(500,input_dim=4,activation='relu'))
# Second hidden layer with 100 neurons
model.add(Dense(100,activation='relu'))
# Third hidden layer with 50 neurons
model.add(Dense(50,activation='relu'))
# Regularization with a Dropout layer to reduce overfitting
# Randomly sets the outgoing edges of hidden units (neurons that make up hidden layers) to 0 at each update of the training phase 
model.add(Dropout(0.3))
# Output layer
model.add(Dense(3, activation='softmax'))

# Since we have multiple classes, we use a loss function that is categorical crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# # Training a model on the training data
test = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=20, batch_size=20, verbose=1)

y_pred = model.predict(x_test)
length = len(y_pred)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

accuracy = np.sum(y_test_class==y_pred_class)/length * 100 
print("\nAccuracy of the dataset", accuracy)

# # We can visualize the accuracy of our model on the 30 testing datapoints (20%)
print(classification_report(y_test_class, y_pred_class))

# # The confusion matrix will show us which species the 30 datapoints best belong to
print(confusion_matrix(y_test_class, y_pred_class))

# Training vs Validation Loss Plot
loss = test.history['loss']
val_loss = test.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T04:16:22.379435Z","iopub.execute_input":"2024-01-22T04:16:22.379918Z","iopub.status.idle":"2024-01-22T04:16:22.838977Z","shell.execute_reply.started":"2024-01-22T04:16:22.379865Z","shell.execute_reply":"2024-01-22T04:16:22.836793Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T04:18:01.907527Z","iopub.execute_input":"2024-01-22T04:18:01.907934Z","iopub.status.idle":"2024-01-22T04:18:01.964860Z","shell.execute_reply.started":"2024-01-22T04:18:01.907902Z","shell.execute_reply":"2024-01-22T04:18:01.963674Z"}}
# Load the dataset
file_path = '/kaggle/input/indian-penal-code-ipc-sections-information/ipc_sections.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to get an overview
df.head()


# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T04:19:43.346750Z","iopub.execute_input":"2024-01-22T04:19:43.347177Z","iopub.status.idle":"2024-01-22T04:19:43.384189Z","shell.execute_reply.started":"2024-01-22T04:19:43.347145Z","shell.execute_reply":"2024-01-22T04:19:43.383029Z"}}
import numpy as np
import pandas as pd

# Load the dataset
file_path = '/kaggle/input/indian-penal-code-ipc-sections-information/ipc_sections.csv'
df = pd.read_csv(file_path)

# Extract relevant columns for input features and target variable
X = df[['Offense', 'Punishment']].values  # Input features
y = df['Description'].values  # Target variable

# Print the first few rows of X and y to verify the data
print("Input Features (X):")
print(X[:5])

print("\nTarget Variable (y):")
print(y[:5])


# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
# Check the data types of 'Offense' and 'Punishment'
print("Data Types:")
print("Offense:", X[:, 0].dtype)
print("Punishment:", X[:, 1].dtype)

# Convert 'Offense' and 'Punishment' to strings if they are not already
X[:, 0] = X[:, 0].astype(str)
X[:, 1] = X[:, 1].astype(str)

# Combine 'Offense' and 'Punishment' into a single text feature
X_text = X[:, 0] + ' ' + X[:, 1]

# Continue with the rest of the code...


# Combine 'Offense' and 'Punishment' into a single text feature
X_text = X[:, 0] + ' ' + X[:, 1]

# Use CountVectorizer to convert text data to numerical format
vectorizer = CountVectorizer()
X_numerical = vectorizer.fit_transform(X_text)

# Use LabelEncoder to convert the target variable to numerical format
label_encoder = LabelEncoder()
y_numerical = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numerical, y_numerical, test_size=0.2, random_state=42)

# Print the shape of the numerical features and target variable
print("Shape of X_numerical:", X_numerical.shape)
print("Shape of y_numerical:", y_numerical.shape)


# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T04:23:20.529569Z","iopub.execute_input":"2024-01-22T04:23:20.530124Z","iopub.status.idle":"2024-01-22T04:23:35.889695Z","shell.execute_reply.started":"2024-01-22T04:23:20.530077Z","shell.execute_reply":"2024-01-22T04:23:35.888416Z"}}
from keras.models import Sequential
from keras.layers import Dense

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y_numerical)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()


# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T04:57:36.837426Z","iopub.execute_input":"2024-01-22T04:57:36.838900Z","iopub.status.idle":"2024-01-22T04:59:00.106334Z","shell.execute_reply.started":"2024-01-22T04:57:36.838820Z","shell.execute_reply":"2024-01-22T04:59:00.105147Z"}}
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder  # Add this import
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load the dataset
file_path = '/kaggle/input/indian-penal-code-ipc-sections-information/ipc_sections.csv'
df = pd.read_csv(file_path)

# Extract relevant columns for input features and target variable
X = df[['Offense', 'Punishment']].astype(str).values  # Convert to string
y = df['Description'].values  # Target variable

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Number of unique classes in your dataset
output_neurons = len(np.unique(y))
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine 'Offense' and 'Punishment' into a single text feature
X_text_train = X_train[:, 0] + ' ' + X_train[:, 1]
X_text_test = X_test[:, 0] + ' ' + X_test[:, 1]

# Use CountVectorizer to convert text data to numerical format
vectorizer = CountVectorizer()
X_train_numerical = vectorizer.fit_transform(X_text_train)
X_test_numerical = vectorizer.transform(X_text_test)

# Compute sample weights for balancing classes
sample_weights = compute_sample_weight('balanced', y_train)

# Define the neural network model with adjustments
model = Sequential()
model.add(Dense(128, input_dim=X_train_numerical.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Set the number of output neurons to match the unique classes
model.add(Dense(output_neurons, activation='softmax'))

# Compile the model with sample weights
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with sample weights
history_weighted = model.fit(X_train_numerical.toarray(), y_train, epochs=1000, batch_size=100, validation_split=0.2, sample_weight=sample_weights)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_numerical.toarray(), y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# %% [code] {"execution":{"iopub.status.busy":"2024-01-22T05:19:03.245731Z","iopub.execute_input":"2024-01-22T05:19:03.246225Z","iopub.status.idle":"2024-01-22T05:19:03.352687Z","shell.execute_reply.started":"2024-01-22T05:19:03.246190Z","shell.execute_reply":"2024-01-22T05:19:03.351561Z"}}
# Your specific input
input_text = "robbery"

# Combine with additional information if needed
# For example, if you have additional context in another variable 'context_text'
# you can concatenate them: input_text = input_text + ' ' + context_text

# Use CountVectorizer to convert the input text to numerical format
input_numerical = vectorizer.transform([input_text])

# Make a prediction using the trained model
prediction = model.predict(input_numerical.toarray())

# If you want the predicted class label (assuming one-hot encoding is used)
predicted_label = np.argmax(prediction)

# If you want to convert the predicted label back to the original label
predicted_label_original = label_encoder.inverse_transform([predicted_label])

# Display or use the prediction as needed
print(predicted_label_original)


# %% [code]

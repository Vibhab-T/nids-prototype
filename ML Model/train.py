from logistic_regression import LogisticRegression

import pandas as pd 
import numpy as np

import joblib #to save the model 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

def main():
    train_df = pd.read_csv("ML Model/KDDTrain+.txt", header=None)
    test_df = pd.read_csv("ML Model/KDDTest+.txt", header=None)


    X_train, y1_train = train_df.iloc[:, :-2], train_df.iloc[:, -2]
    X_test, y1_test = test_df.iloc[:, :-2], test_df.iloc[:, -2]

    # In the dataset, the second last column is categorical but has multiple categories.
    # First we convert this multiple categoires into binary categoreis: "normal" and "anomaly{neptrune, smurf, etc}"
    y1_train_binary = y1_train.apply(lambda x: 'normal' if x == 'normal.' else 'anomaly')
    y1_test_binary = y1_test.apply(lambda x: 'normal' if x == 'normal.' else 'anomaly')

    # Label Encoding the target variables 
    le = LabelEncoder()
    le.fit(y1_train_binary)
    y1_train_encoded = le.transform(y1_train_binary)
    y1_test_encoded = le.transform(y1_test_binary)

    # One Hot Enocding Categorical Features. Columns 2 to 4. Index 1 to 3.
    cat_col = [1, 2, 3]
    num_col = [i for i in range(X_train.shape[1]) if i not in cat_col]

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_col), #standerizes numerical columns
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)   #one-hot encodes categorical columns
        ],
        remainder='passthrough'
    )

    column_transformer.fit(X_train)  #fits the transformer to the training data
    X_train_encoded = column_transformer.transform(X_train) # transforms the training data
    X_test_encoded = column_transformer.transform(X_test)   # transforms the test data

    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(lr=0.01, n_iter=1000)
    log_reg.fit(X_train_encoded, y1_train_encoded)

    # Make predictions on the test set
    y1_pred = log_reg.predict(X_test_encoded)

    # Calculate accuracy
    accuracy = accuracy_score(y1_test_encoded, y1_pred)
    print(accuracy)

    print(classification_report(y1_test_encoded, y1_pred, target_names=le.classes_))

    #save the model 
    joblib.dump(log_reg, 'ML Model/logistic_regression_model.pkl') 
    
    #save the column transformer and label encoder
    #so that we can use them later to transform new data
    joblib.dump(column_transformer, 'ML Model/column_transformer.pkl')
    joblib.dump(le, 'ML Model/label_encoder.pkl') 
  
main()

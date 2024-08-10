import numpy as np
import pandas as pd
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# # # Importing the dataset
# df = pd.read_csv(r'F:\Level_4_Term_2\CSE_472\ml\1805064\TelcoCustomerChurn.csv')

# print(df.describe)

SEED= 40

def print_unique_values(dataframe):
    """
    Print unique values in each column of a Pandas DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    """
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        print(f"Column: {column}")
        print("Unique Values:")
        print(unique_values)
        print("\n" + "-"*30 + "\n")

def dataset1():
    df = pd.read_csv(r'F:\Level_4_Term_2\CSE_472\ml\1805064\TelcoCustomerChurn.csv')
    df.drop(["customerID"], inplace=True, axis=1)
    df["MultipleLines"].replace({"No phone service": "No"}, inplace=True)


    for key in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        df[key].replace({"No internet service": "No"}, inplace=True)
    df["Churn"].replace({"Yes": 1, "No": 0}, inplace=True)
    df["gender"].replace({"Female": 0, "Male": 1}, inplace=True)
    for key in [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ]:
        df[key].replace({"Yes": 1, "No": 0}, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Split into Features and Target
    X = df.drop("Churn", axis=1)  # Features
    y = df["Churn"]  # Target variable

    # Split into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    
    # Handle Missing Values in Training Set
    # Example: Impute missing values with the mean of 'TotalCharges'
    imputer = SimpleImputer(strategy="mean")
    # Fit the imputer on the training set to calculate the mean
    imputer.fit(X_train[["TotalCharges"]])
    # Get the mean value used for imputation
    mean_value = imputer.statistics_[0]
    X_train["TotalCharges"] = imputer.transform(X_train[["TotalCharges"]])
    # Use the mean from the training set to impute missing values in the testing set
    X_test["TotalCharges"] = imputer.transform(X_test[["TotalCharges"]])

    # Normalize numerical features
    scaler = StandardScaler()
    for key in ["tenure", "MonthlyCharges", "TotalCharges"]:
        X_train[key] = scaler.fit_transform(X_train[[key]])
        X_test[key] = scaler.transform(X_test[[key]])

    # Encoding categorical features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def dataset2():
    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    # Load training data
    df_train = pd.read_csv(url_train, names=column_names, sep=',\s', na_values=["?"], engine='python')

    # Load testing data
    df_test = pd.read_csv(url_test, names=column_names, sep=',\s', na_values=["?"], engine='python', skiprows=1)  # skiprows=1 to skip the first row in the test set
    missing_columns= ["workclass", "occupation", "native-country"]

    # Replace missing values in training data
    for column in missing_columns:
        most_frequent_category = df_train[column].mode()[0]
        df_train[column].fillna(most_frequent_category, inplace=True)
        df_test[column].fillna(most_frequent_category, inplace= True)

    # Replace 'sex' values
    sex_mapping = {'Male': 1, 'Female': 0}
    df_train['sex'] = df_train['sex'].replace(sex_mapping)
    df_test['sex'] = df_test['sex'].replace(sex_mapping)


    # Replace 'income' values
    income_mapping_test = {'<=50K.': 0, '>50K.': 1}
    income_mapping = {'<=50K': 0, '>50K': 1}
    df_train['income'] = df_train['income'].replace(income_mapping)
    df_test['income'] = df_test['income'].replace(income_mapping_test)

    # Split the data into features (X) and target variable (y)
    X_train = df_train.drop("income", axis=1)
    y_train = df_train["income"]

    X_test = df_test.drop("income", axis=1)
    y_test = df_test["income"]


    #Normalize

    scaler = StandardScaler()
    for key in ["age", 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        X_train[key] = scaler.fit_transform(X_train[[key]])
        X_test[key] = scaler.transform(X_test[[key]])

        # Encoding categorical features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train= X_train.drop('native-country_Holand-Netherlands', axis= 1)


    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()



def dataset3():
  df = pd.read_csv(r'F:\Level_4_Term_2\CSE_472\ml\1805064\creditcard.csv')

  df= df.drop('Time', axis= 1)

  # Separate positive and negative samples
  positive_samples = df[df['Class'] == 1]
  negative_samples = df[df['Class'] == 0].sample(n=20000, random_state=SEED)

  # Concatenate positive and negative samples to create the smaller subset
  subset_df = pd.concat([positive_samples, negative_samples])

  # Split the subset into features (X) and target variable (y)
  X = subset_df.drop('Class', axis=1)
  y = subset_df['Class']

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

  return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


#write a function for sigmoid
def sigmoid(x):
    return 1/(1+ np.exp(-x))


#write mean square error function
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


#write a function for gradient descent
def gradient_descent(x, y_true, y_pred):
    n = x.shape[0]
    gradient= np.dot(x.T, (y_true- y_pred)* y_pred* (1- y_pred))/ n
    return gradient

def normalize(X):
    # standardizing to have zero mean and unit variance
    X= X.astype('float64')

    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def top_k_information_gain(X_train, y_train, k):
    # Apply SelectKBest with f_classif for feature selection
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Display the selected features
    print("Selected Features Indices:", selector.get_support(indices=True))

    # If you want to get a numpy array with only the selected columns' data
    X_train_selected_array = X_train[:, selector.get_support(indices=True)]

    # Display the shape of the selected array
    print("Shape of Selected Array:", X_train_selected_array.shape)
    return X_train_selected_array, selector.get_support(indices=True)

def logistic_regression_train(X, y_true, epochs,k=None,  lr= 0.01, threshold=0):
    # Getting number of features.
    no_samples, no_features = X.shape

    # Initializing weights
    if k is None:
      w = np.zeros((no_features + 1, 1))
    else:
      w = np.zeros((k + 1, 1))

    X= normalize(X)

    # Adding bias term to features.
    X = np.concatenate((X, np.ones((no_samples, 1))), axis=1)

    y_true = y_true.reshape(no_samples, 1)


    # Training loop.
    for epoch in range(epochs):
        # Calculating hypothesis/prediction.
        y_hat = sigmoid(np.dot(X, w))

        # Getting the gradients of loss w.r.t parameters.
        dw = gradient_descent(X, y_true, y_hat)
        dw = dw.astype(np.float64)

        # Updating the parameters.
        w += lr * dw

        l = mse(y_true, y_hat)
        if l < threshold:
            # print("loss found less than 0.5.loss: ", l) # for weak learner threshold needs to be set
            break

    return w

#write a function for logistic regression prediction
def logistic_regression_predict(X, w):
    # Getting number of features.
    no_samples, no_features = X.shape

    X= normalize(X)

    # Adding bias term to features.
    X = np.concatenate((X, np.ones((no_samples, 1))), axis=1)

    y_pred = sigmoid(np.dot(X, w))

    predictions_list = [1 if i >= 0.5 else 0 for i in y_pred]

    return np.array(predictions_list).reshape(no_samples, 1)

#function for performance measure

def performance(y_true, y_pred):
    Accuracy = np.sum(y_true == y_pred) / len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    PPV = tp / (tp + fp)
    FDR = fp / (tp + fp)
    F1 = 2 * ((PPV * TPR) / (PPV + TPR))
    #print accuracy with 2 digit precision
    print("Accuracy: ", round(Accuracy* 100, 2), "%")
    #print other performance measures with 4 digit precision

    print("True Positive Rate: ", round(TPR, 4))
    print("True Negative Rate: ", round(TNR, 4))
    print("Positive Predictive Value: ", round(PPV, 4))
    print("False Discovery Rate: ", round(FDR, 4))
    print("F1 Score: ", round(F1, 4))






def adaboost(X, y_true, K):
    # examples : set of N labled examples
    # weak_learner: a learning algo
    # K: number of hypothesis in the ensemble

    # sample_size = N
    sample_size, num_features = X.shape
    epsilon = np.finfo(float).eps  # to avoid division by 0

    # w: a vector of N example weights, initially 1/N
    # h: a vector of K hypothesis
    # z: a vector of K hypothesis weights
    w = np.full(sample_size, 1 / sample_size)
    h = []
    z = []


    for k in range(K):
        print("k round: ", k+1)

        examples = np.concatenate((X, y_true), axis=1)

        np.random.seed(k)
        data = examples[np.random.choice(sample_size, size=sample_size, replace=True, p=w)]
        data_X = data[:, :num_features]
        data_y_true = data[:, -1:]

        weight= logistic_regression_train(data_X, data_y_true, epochs=1000, lr=0.05, threshold= .2)

        h.append(weight)
        error = 0

        pred = logistic_regression_predict(X, weight)

        for i in range(sample_size):
            if pred[i] != y_true[i]:
                error += w[i]

        # use 2 digit precision
        error = round(error, 2)
        # print("k: ", k+1, "    error: ", error)
        if error > 0.55:
            continue

        
        for i in range(sample_size):
            if pred[i] == y_true[i]:
                w[i] = (w[i] * error) / (1 - error)

        # normalize weights
        w = w / np.sum(w)

        z.append(math.log((1 - error) / (error + epsilon), 2))

    return h, np.array(z).reshape(len(h), 1)




def adaboost_preduct_performance(dataset, y_true, hyp_vectors, z_values):
    num_samples = dataset.shape[0]
    num_hypotheses = len(hyp_vectors)

    # normalizing inputs X
    dataset = normalize(dataset)

    # augmenting dummy input attribute 1 to each row of X
    dataset = np.concatenate((dataset, np.ones((num_samples, 1))), axis=1)

    # calculating hypotheses
    y_predicteds = []

    for i in range(num_hypotheses):
        #use logistic regression predict
        y_predicted = sigmoid(np.dot(dataset, hyp_vectors[i]))
        y_predicteds.append([1 if y_pred >= 0.5 else -1 for y_pred in y_predicted])

    y_predicteds = np.array(y_predicteds)

    # calculating weighted majority hypothesis and storing predictions
    weighted_majority_hypothesis = np.dot(y_predicteds.T, z_values)
    predictions = [1 if y_pred >= 0 else 0 for y_pred in weighted_majority_hypothesis]

    predictions = np.array(predictions).reshape(num_samples, 1)
    Accuracy = np.sum(y_true == predictions) / len(y_true)
    #print accuracy with 2 digit precision
    print("Accuracy: ", round(Accuracy* 100, 2), "%")

##############################################################################################################
#DATA PREPROCESSING
##############################################################################################################

#call dataset1() function
#Telco Customer Churn dataset
(churn_train, churn_test, churn_target_train, churn_target_test)= dataset1()


churn_target_train= churn_target_train.reshape(churn_target_train.shape[0], 1)
churn_target_test= churn_target_test.reshape(churn_target_test.shape[0], 1)

########for top k features########

# churn_train, indices= top_k_information_gain(churn_train, churn_target_train, 5)
# churn_test= churn_test[:, indices]

#call dataset2() function
#Adult dataset
# (adult_train, adult_test, adult_target_train, adult_target_test)= dataset2()

# adult_target_train= adult_target_train.reshape(adult_target_train.shape[0], 1)
# adult_target_test= adult_target_test.reshape(adult_target_test.shape[0], 1)

# #######for top k features#######

# adult_train, indices= top_k_information_gain(adult_train, adult_target_train, 15)
# adult_test= adult_test[:, indices]


#call dataset3() function
#Credit card dataset
# (credit_train, credit_test, credit_target_train, credit_target_test)= dataset3()


# credit_target_train= credit_target_train.reshape(credit_target_train.shape[0], 1)
# credit_target_test= credit_target_test.reshape(credit_target_test.shape[0], 1)

########for top k features########

# credit_train, indices= top_k_information_gain(credit_train, credit_target_train, 5)
# credit_test= credit_test[:, indices]

##############################################################################################################
#LOGISTIC REGRESSION
##############################################################################################################

#For dataset1
w= logistic_regression_train(churn_train, churn_target_train, epochs=1000, lr=0.01, threshold= .2)

print("Logistic regression with sigmoid for dataset1: Train")
performance(churn_target_train, logistic_regression_predict(churn_train, w))

print("Logistic regression with sigmoid for dataset1: Test")
performance(churn_target_test, logistic_regression_predict(churn_test, w))

#For dataset2
# w= logistic_regression_train(adult_train, adult_target_train, k= 15, epochs=1000, lr=0.01, threshold= .2)

# print("Logistic regression with sigmoid for dataset2: Train")
# performance(adult_target_train, logistic_regression_predict(adult_train, w))

# print("Logistic regression with sigmoid for dataset2: Test")
# performance(adult_target_test, logistic_regression_predict(adult_test, w))

#For dataset3
# w= logistic_regression_train(credit_train, credit_target_train, epochs=1000, lr=0.01, threshold= .2)

# print("Logistic regression with sigmoid for dataset3: Train")
# performance(credit_target_train, logistic_regression_predict(credit_train, w))

# print("Logistic regression with sigmoid for dataset3: Test")
# performance(credit_target_test, logistic_regression_predict(credit_test, w))


##############################################################################################################
#ADABOOST
##############################################################################################################

#For dataset1
print("\n\nAdaboost Result for telco customer churn dataset:")
K_values = [5, 10, 15, 20]
for k in K_values:
    print("\nK :", k)
    print("Training set:")
    hypothesis, hyp_weights = adaboost(churn_train, churn_target_train, K= k)
    adaboost_preduct_performance(churn_train, churn_target_train, hypothesis, hyp_weights)
    print()
    print("Test set: ")
    adaboost_preduct_performance(churn_test, churn_target_test, hypothesis, hyp_weights)

#For dataset2
# print("\n\nAdaboost Result for adult dataset:")
# K_values = [5, 10, 15, 20]
# for k in K_values:
#     print("\nK :", k)
#     print("Training set:")
#     hypothesis, hyp_weights = adaboost(adult_train, adult_target_train, K= k)
#     adaboost_preduct_performance(adult_train, adult_target_train, hypothesis, hyp_weights)
#     print()
#     print("Test set: ")
#     adaboost_preduct_performance(adult_test, adult_target_test, hypothesis, hyp_weights)

#For dataset3
# print("\n\nAdaboost Result for credit card dataset:")
# K_values = [5, 10, 15, 20]
# for k in K_values:
#     print("\nK :", k)
#     print("Training set:")
#     hypothesis, hyp_weights = adaboost(credit_train, credit_target_train, K= k)
#     adaboost_preduct_performance(credit_train, credit_target_train, hypothesis, hyp_weights)
#     print()
#     print("Test set: ")
#     adaboost_preduct_performance(credit_test, credit_target_test, hypothesis, hyp_weights)

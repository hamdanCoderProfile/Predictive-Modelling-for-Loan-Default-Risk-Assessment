# Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the Data

df= pd.read_csv('/content/Loan_default.csv')

# Data Exploration and Preprocessing

df.head()

## Dropping the column - 'LoanID'

df = df.drop(['LoanID'], axis=1)

## Checking Data Types

df.info()

## Checking for null values in dataset

df.isnull().sum()

df.describe()

## Checking values in categorical columns

df['Education'].value_counts()

df['EmploymentType'].value_counts()

df['MaritalStatus'].value_counts()

df['HasMortgage'].value_counts()

df['HasDependents'].value_counts()

df['LoanPurpose'].value_counts()

df['HasCoSigner'].value_counts()

df['Default'].value_counts()

## Converting categorical columns into numerical data

from sklearn.preprocessing import LabelEncoder

# List of categorical columns to be label encoded
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate through each categorical column and apply label encoding
for col in categorical_columns:
df[col] = label_encoder.fit_transform(df[col])


# Drop the original categorical columns if needed
# df.drop(columns=categorical_columns, inplace=True)

## Checking relation between columns in dataset

correlation_with_default = df.corr()['Default'].sort_values(ascending=False)

print(correlation_with_default)

## Checking for collinearity in data

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Create a DataFrame containing only the independent variables (features)
X = df.drop(columns=['Default'])

# Add a constant to the independent variables matrix for intercept calculation
X = add_constant(X)

# Calculate VIF for each independent variable
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print VIF values
print(vif_data)


## Correcting imbalance in dataset and removing outliers from data

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

# Assuming your dataset is stored in a DataFrame called 'df'

# Step 1: Balance the dataset using SMOTE
X = df.drop('Default', axis=1) # Features
y = df['Default'] # Target

# Instantiate SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Default'])], axis=1)

# Step 2: Remove outliers using Isolation Forest
# Assuming your dataset is already scaled appropriately

# Instantiate Isolation Forest
isolation_forest = IsolationForest(contamination=0.1, random_state=42)

# Fit Isolation Forest
outlier_preds = isolation_forest.fit_predict(df_resampled.drop('Default', axis=1))

# Filter outliers
df_no_outliers = df_resampled[outlier_preds != -1]

# Separate target variable from features
X_no_outliers = df_no_outliers.drop('Default', axis=1)
y_no_outliers = df_no_outliers['Default']

# Concatenate features and target variable
df_final = pd.concat([X_no_outliers, y_no_outliers], axis=1)

# Now, df_final contains your balanced dataset without outliers, including the target variable


df_final['Default'].value_counts()

## Normalizing the dataset

from sklearn.preprocessing import StandardScaler

# Assuming df_final is your final dataset

# Columns to be standardized
columns_to_standardize = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate',
'LoanTerm', 'DTIRatio']

# Instantiate StandardScaler
scaler = StandardScaler()

# Standardize selected columns
df_final[columns_to_standardize] = scaler.fit_transform(df_final[columns_to_standardize])

# Now, df_final contains standardized numerical columns


## Dividing the data

from sklearn.model_selection import train_test_split

# Assuming df_final is your final dataset

# Splitting into features (X) and target variable (y)
X = df_final.drop('Default', axis=1)
y = df_final['Default']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, you have X_train (features for training), X_test (features for testing), y_train (target variable for training), and y_test (target variable for testing)


# Training the model

import numpy as np

class LogisticRegressionWithRegularization:
def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_val=0.01):
self.learning_rate = learning_rate
self.num_iterations = num_iterations
self.lambda_val = lambda_val
self.weights = None
self.bias = None

def sigmoid(self, z):
return 1 / (1 + np.exp(-z))

def fit(self, X, y):
num_samples, num_features = X.shape
self.weights = np.zeros(num_features)
self.bias = 0

# gradient descent
for _ in range(self.num_iterations):
# linear model
linear_model = np.dot(X, self.weights) + self.bias
# sigmoid function
y_predicted = self.sigmoid(linear_model)

# compute gradients with regularization
dw = (1 / num_samples) * (np.dot(X.T, (y_predicted - y)) + 2 * self.lambda_val * self.weights)
db = (1 / num_samples) * np.sum(y_predicted - y)

# update parameters
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db

def predict(self, X):
linear_model = np.dot(X, self.weights) + self.bias
y_predicted = self.sigmoid(linear_model)
y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
return y_predicted_cls


log_reg = LogisticRegressionWithRegularization()
log_reg.fit(X_train, y_train)

# Hypothesis Testing

# Define hypothesis testing function
def wald_test(model, X, y):
# Get coefficient estimates and their standard errors
coef = model.weights
num_samples, num_features = X.shape
y_predicted = model.predict(X)
residuals = y_predicted - y
sigma_squared = np.dot(residuals, residuals) / (num_samples - num_features - 1)
cov_matrix = np.linalg.inv(np.dot(X.T, X)) * sigma_squared

# Compute z-statistics
z_stat = coef / np.sqrt(np.diag(cov_matrix))

# Compute Wald statistic
wald_stat = z_stat ** 2

# Compute p-values
p_values = 1 - chi2.cdf(wald_stat, df=1)

return {'Coefficient': coef.flatten(), 'Standard Error': np.sqrt(np.diag(cov_matrix)), 'Z-Statistic': z_stat, 'Wald Statistic': wald_stat, 'P-Value': p_values}


# Perform hypothesis testing
results = wald_test(log_reg, X_train, y_train)

# Print results
print("Hypothesis Testing Results:")
print("{:<20} {:<20} {:<20} {:<20} {:<20}".format('Feature', 'Coefficient', 'Standard Error', 'Z-Statistic', 'P-Value'))
for i in range(len(log_reg.weights)):
print("{:<20} {:<20} {:<20} {:<20} {:<20}".format(f'Feature {i}', results['Coefficient'][i], results['Standard Error'][i], results['Z-Statistic'][i], results['P-Value'][i]))

# Testing effectiveness of the model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Predict on test data
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# Generate classification report
class_report = classification_report(y_test, y_pred)

# Print the classification report
print("Classification Report:")
print(class_report)

# Exploring top 3 resons for prediction along with its values

# Assuming df_final is your final dataset and y_pred contains your predictions

# Define feature names (replace these with your actual feature names)
feature_names = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Initialize an empty DataFrame to store top reasons
top_reasons_df = pd.DataFrame(columns=['Prediction', 'Top Reason 1', 'Value 1', 'Top Reason 2', 'Value 2', 'Top Reason 3', 'Value 3'])

# Iterate through the last 10 predictions and extract top three reasons for each
for i, prediction in enumerate(y_pred[-10:], start=len(y_pred)-10):
# Get coefficients from the logistic regression model for this prediction
coefficients = log_reg.weights

# Create a dictionary to map feature names to coefficients
feature_coefficients = dict(zip(feature_names, coefficients))

# Sort the features based on their coefficients
sorted_features = sorted(feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

# Extract top three reasons
top_three_reasons = sorted_features[:3]

# Extract top three reasons and their corresponding values
top_three_reasons_values = [(feature, coefficient, df_final.iloc[i][feature]) for feature, coefficient in top_three_reasons]

# Store top three reasons in DataFrame
row_values = [prediction]
for j in range(3):
if j < len(top_three_reasons_values):
reason, coefficient, value = top_three_reasons_values[j]
row_values.extend([reason, value])
else:
row_values.extend(['', ''])
top_reasons_df.loc[i] = row_values

# Print or use top_reasons_df as needed
print(top_reasons_df)
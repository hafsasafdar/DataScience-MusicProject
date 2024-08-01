#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:15:45 2023

@author: hafsasafdar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Sat Dec  9 10:18:19 2023

#@author: hafsasafdar


import os
print(os.getcwd())  # Display current working directory
os.chdir('/Users/cheema/Downloads')  # Change working directory if needed # Replace with your actual file name # Should print True if the file is in the working directory


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import permutation_test, norm, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, silhouette_samples
from scipy.stats import shapiro
import random
from scipy.stats import ttest_ind

file_path = 'spotify52kData.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

mySeed = 4585 #id number
random.seed(mySeed)

#EDA to get to know the distribution of all of these factors
data.hist(bins=20, figsize=(15, 10))
plt.show()
data.shape

# Categorical features distribution
sns.countplot(x='explicit', data=data)
sns.countplot(x='mode', data=data)
plt.show()

#Description of numerical varaibles in the data
description = data.describe()
dataTypes = data.dtypes

numeric_features = data[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'popularity', 'instrumentalness', 'liveness', 'valence', 'tempo']].dropna()

numeric_factor_names = numeric_features.columns

# Correlation matrix
correlation_matrix = np.corrcoef(numeric_features, rowvar=False)
plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, cmap='RdYlBu_r', interpolation='nearest')  # Use 'RdYlBu_r' colormap
plt.colorbar()
plt.xticks(np.arange(len(numeric_factor_names)), numeric_factor_names, rotation=45)
plt.yticks(np.arange(len(numeric_factor_names)), numeric_factor_names)
plt.title('Correlation Matrix of Numeric Features')
plt.show()


pca_subset = data[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].dropna()
pca_subset.shape
pca_subset = pca_subset.reset_index(drop=True)
factor_names = pca_subset.columns

#Correlation Matrix
# Doing this, w ecan see that energy and loudness are positively correlated
# Energy and acousticness, Loudness and acousticness are highly negatively correlated
#Others are generally low correlations.

#Do the PCA do identify some indeopendent features among thes ecorrelated factors
zscoredData = stats.zscore(pca_subset)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing
# Rotated Data:
rotatedData = pca.fit_transform(zscoredData)

varExplained = eigVals/sum(eigVals)*100
for ii in range(len(varExplained)):
   print(varExplained[ii].round(3))
#We can note that there are some factors which account fopr much more varaince then others. Some account for as much
# as 23%, 13% etc. The question is which ones we will pick?

#%% 5) Making a scree plot

num_components = len(eigVals)
x = np.linspace(1, num_components, num_components)
plt.bar(x, eigVals, color='gray')
plt.plot([0, num_components + 1], [1, 1], color='orange')  # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.show()

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

#%%  Interpreting the factors
for whichPrincipalComponent in range(3):
   plt.figure()
   plt.bar(range(len(factor_names)), pca.components_[whichPrincipalComponent, :] * -1)  # note: multiplied by -1 for direction
   plt.xlabel('Factor Names')
   plt.ylabel('Loading')
   plt.title(f'Loading Plot for Principal Component {whichPrincipalComponent}')
   plt.xticks(range(len(factor_names)), factor_names, rotation=45)  # Setting tick labels
   plt.show()



#  The actual clustering using K-Means
x = rotatedData

# 2i) Determine the optimal number of clusters using silhouette analysis
numClusters = 4
sSum = np.empty([numClusters, 1]) * np.NaN
print(sSum)

for ii in range(2, numClusters + 2):
   print(ii)
   kMeans = KMeans(n_clusters=int(ii), n_init=10).fit(x)
   cId = kMeans.labels_
   s = silhouette_samples(x, cId)
   sSum[ii-2] = sum(s)

# Plot the sum of the silhouette scores as a function of the number of clusters
plt.plot(np.arange(2, numClusters + 2), sSum)  # Use np.arange instead of np.linspace
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

optimal_num_clusters = np.argmax(sSum) + 2  # 2-based indexing
print(f"Optimal number of clusters: {optimal_num_clusters}")

optimal_num_clusters = 2 # 2-based indexing
kMeans = KMeans(n_clusters=optimal_num_clusters, n_init=10).fit(x)
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_

# Plot the color-coded data:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for ii in range(optimal_num_clusters):
   plotIndex = np.argwhere(cId == int(ii))
   ax.scatter(x[plotIndex, 0], x[plotIndex, 1], x[plotIndex, 2], s=1)
   ax.scatter(cCoords[int(ii), 0], cCoords[int(ii), 1], cCoords[int(ii), 2], s=50, color='black')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()



##%% Q1
dataTypes = data.dtypes

dataQ1 = data[['danceability']].copy()
danceability_data = dataQ1['danceability'].dropna()

# Plot the histogram
sns.histplot(danceability_data, bins=20, kde=True, color='skyblue', edgecolor='black')

# Overlay a normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, danceability_data.mean(), danceability_data.std())
plt.plot(x, p, 'k', linewidth=2)

plt.title('Distribution of Danceability with Normal Distribution Overlay')
plt.xlabel('Danceability')
plt.ylabel('Frequency')

plt.show()


#Q2

# Assuming 'duration' and 'popularity' are columns in your DataFrame (e.g., data)
dataQ2 = data[['duration', 'popularity']]
print(dataQ2.isnull().any());
plt.scatter(dataQ2['duration'], dataQ2['popularity'])
plt.title('Scatterplot of Song Duration vs Popularity')
plt.xlabel('Duration (m)')
plt.ylabel('Popularity')
plt.show()

spearman_coefficient, _ = scipy.stats.spearmanr(dataQ2['duration'], dataQ2['popularity'])

print(f"Spearman Correlation Coefficient: {spearman_coefficient}")

# Q3
dataQ3 = data[['explicit', 'popularity']]
print(dataQ3.isnull().any());

# Separate data for explicit and non-explicit songs
explicit_data = dataQ3[dataQ3['explicit'] == True]['popularity']
non_explicit_data = dataQ3[dataQ3['explicit'] == False]['popularity']

# Create histograms to see
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(explicit_data, bins=20, color='blue', alpha=0.7)
plt.title('Popularity Distribution for Explicit Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(non_explicit_data, bins=20, color='green', alpha=0.7)
plt.title('Popularity Distribution for Non-Explicit Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Define function to calculate confidence interval
def confidence_interval(data, alpha=0.05):
 n = len(data)
 mean = np.mean(data)
 std = np.std(data)
 se = std / np.sqrt(n)
 margin_of_error = stats.norm.ppf(1 - alpha / 2) * se
 lower_bound = mean - margin_of_error
 upper_bound = mean + margin_of_error
 return lower_bound, upper_bound

num_permutations = 1000
num_bins = 30

# Calculate confidence intervals for both groups
explicit_ci = confidence_interval(explicit_data)
non_explicit_ci = confidence_interval(non_explicit_data)

# Print the results
print("Confidence interval for Explicit Songs:", explicit_ci)
print("Confidence interval for Non-Explicit Songs:", non_explicit_ci)

# Define a statistic function
def custom_statistic(x, y):
   return np.mean(x) - np.mean(y)

# Perform permutation test
# Perform permutation test for explicit data

permuted_stats_explicit = np.array([custom_statistic(np.random.permutation(np.concatenate([explicit_data, non_explicit_data]))[:len(explicit_data)],
                                                     np.random.permutation(np.concatenate([explicit_data, non_explicit_data]))[len(explicit_data):])
                                    for _ in range(num_permutations)])

# Calculate the observed statistic
observed_statistic_explicit = custom_statistic(explicit_data, non_explicit_data)

# Calculate p-value for explicit data
p_value_explicit = (np.sum(permuted_stats_explicit >= observed_statistic_explicit) + 1) / (num_permutations + 1)

# Check for statistical significance and print results
if p_value_explicit < 0.05:
   print("The p-value is less than 0.05, indicating that the observed difference is statistically significant.")
   print("Explicit songs are more popular on average.")
else:
   print("The p-value is greater than or equal to 0.05, suggesting that the observed difference could be due to random chance.")
   print("There is no significant evidence that explicit songs are more popular on average.")

# Plot the distribution of permuted statistics for explicit data

plt.hist(permuted_stats_explicit, bins=num_bins, alpha=0.7, label='Permuted Distribution (Explicit)')

# Plot a vertical line for the observed statistic
plt.axvline(x=observed_statistic_explicit, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic (Explicit)')
plt.title('Permutation Test Distribution (Explicit)')
plt.xlabel('Difference in Means (Permuted Statistics)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Q4 Q4 Q4
dataQ4 = data[['popularity', 'mode']].copy()

plt.figure(figsize=(10, 6))

# Major key (mode = 1)
plt.subplot(1, 2, 1)
plt.hist(dataQ4[dataQ4['mode'] == 1]['popularity'], bins=20, color='blue', alpha=0.7)
plt.title('Popularity Distribution for Major Key Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')

# Minor key (mode = 0)
plt.subplot(1, 2, 2)
plt.hist(dataQ4[dataQ4['mode'] == 0]['popularity'], bins=20, color='green', alpha=0.7)
plt.title('Popularity Distribution for Minor Key Songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


major_key_data = dataQ4[dataQ4['mode'] == 1]['popularity']
minor_key_data = dataQ4[dataQ4['mode'] == 0]['popularity']

# Calculate confidence intervals for both groups
major_ci = confidence_interval(major_key_data)
minor_ci = confidence_interval(minor_key_data)
print("Confidence interval for Major Key Songs:", major_ci)
print("Confidence interval for Minor Key Songs:", minor_ci)

print(custom_statistic(major_key_data, minor_key_data))

# Perform permutation test

permuted_stats_key = np.array([custom_statistic(np.random.permutation(np.concatenate([major_key_data, minor_key_data]))[:len(major_key_data)],
                                                np.random.permutation(np.concatenate([major_key_data, minor_key_data]))[len(major_key_data):])
                              for _ in range(num_permutations)])
observed_statistic_key = custom_statistic(major_key_data, minor_key_data)
p_value_key = (np.sum(permuted_stats_key >= observed_statistic_key) + 1) / (num_permutations + 1)

# Display the results for key data
print("P_Value of key is " , p_value_key)
if p_value_key < 0.05:
   print("Songs in major key are more popular on average.")
else:
   print("There is no statistiaclly significant difference in the popularity betweene the two groups")

# Plot the distribution of permuted statistics for key data
plt.hist(permuted_stats_key, bins=num_bins, alpha=0.7, label='Permuted Distribution (Key)')
plt.axvline(x=observed_statistic_key, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic (Key)')
plt.title('Permutation Test Distribution (Key)')
plt.xlabel('Difference in Means (Permuted Statistics)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#Q5

dataQ5 = data[['energy', 'loudness']]  # Fix the column selection
plt.figure(figsize=(10, 6))
plt.scatter(dataQ5['energy'], dataQ5['loudness'], alpha=0.5)
plt.title('Scatterplot of Loudness vs. Energy')
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.show()

correlation, _ = spearmanr(dataQ5['energy'], dataQ5['loudness'])
print(f"Spearman Rank Correlation Coefficient: {correlation:.4f}")

#Q6
features = ["duration", "danceability", "energy", "loudness", "speechiness",
           "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
target = "popularity"


# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.suptitle('Scatter Plots of Song Features vs. Popularity', fontsize=16)

for i, feature in enumerate(features):
   row = i // 4
   col = i % 4

   # Scatter plot
   axes[row, col].scatter(data[feature], data['popularity'], alpha=0.25)
   axes[row, col].set_title(feature)
   axes[row, col].set_xlabel(feature)
   axes[row, col].set_ylabel('Popularity')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Set a seaborn style for better aesthetics
sns.set(style="whitegrid")
scatter_matrix = sns.pairplot(data[features + ['popularity']], height=2, aspect=1.5)
scatter_matrix.fig.suptitle('Scatter Plots of Song Features vs. Popularity', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# Placeholder for results
results = []
scaler = StandardScaler()

# Initialize Ridge model
ridge = Ridge(alpha=1.0)  # You can adjust the alpha parameter

best_r2 = 0
best_r2_feature = None
best_rmse = float('inf')
best_rmse_feature = None

# Iterate through each feature
for feature in features:
   data_subset = data[[feature, target]].copy()

   features_scaled = pd.DataFrame(scaler.fit_transform(data_subset[[feature]]), columns=[feature])

   X_train, X_test, y_train, y_test = train_test_split(features_scaled, data_subset[target], test_size=0.2, random_state=42)

   ridge.fit(X_train, y_train)  # Fit the Ridge model

   y_pred = ridge.predict(X_test)  # Predict on the test set

   r2 = r2_score(y_test, y_pred)   # Calculate R-squared

   rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate RMSE

   # Check if current feature has the best R-squared
   if r2 > best_r2:
       best_r2 = r2
       best_r2_feature = feature

   # Check if current feature has the least RMSE
   if rmse < best_rmse:
       best_rmse = rmse
       best_rmse_feature = feature

   # Get coefficients
   coefficients = ridge.coef_

   results.append({
       'Feature': feature,
       'R-squared': r2,
       'RMSE': rmse,
       'Coefficients': coefficients
   })

# Print the results
for result in results:
   print(f"\nFeature: {result['Feature']}")
   print(f"R-squared: {result['R-squared']:.4f}")
   print(f"RMSE: {result['RMSE']:.4f}")
   print(f"Coefficients: {result['Coefficients']}")

print(f"\nFeature with the Best R-squared: {best_r2_feature} (R-squared: {best_r2:.4f})")
print(f"Feature with the Least RMSE: {best_rmse_feature} (RMSE: {best_rmse:.4f})")

# PLOTTIG THE BEST ONE
selected_feature = best_r2_feature

selected_feature_data = data[[selected_feature, target]].copy()

selected_feature_scaled = pd.DataFrame(scaler.fit_transform(selected_feature_data[[selected_feature]]), columns=[selected_feature])
X_train, X_test, y_train, y_test = train_test_split(selected_feature_scaled, selected_feature_data[target], test_size=0.2, random_state=42)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(selected_feature_scaled, selected_feature_data[target], alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Ridge Regression')
plt.title(f'Scatter Plot of {selected_feature} vs Popularity with Ridge Regression')
plt.xlabel(selected_feature)
plt.ylabel('Popularity')
plt.legend()
plt.show()


# Q7
# Scatter plot of instrumentalness vs popularity
plt.figure(figsize=(10, 6))
plt.scatter(data['instrumentalness'], data['popularity'], alpha=0.5)
plt.title('Scatter Plot of Instrumentalness vs Popularity')
plt.xlabel('Instrumentalness')
plt.ylabel('Popularity')
plt.show()

# Subset the data
dataQ7 = data[features + [target]].dropna() #features and targe defined in Q6

# Separate features and target variable
X_all_features = dataQ7[features]
y_all_predictors = dataQ7[target]

# Scale features
scaler = StandardScaler()
X_scaled_all_features = scaler.fit_transform(X_all_features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_all_features, y_all_predictors, test_size=0.2, random_state=42)

# Initialize Ridge model
ridge = Ridge(alpha=1.0)  # You can adjust the alpha parameter
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Coefficients: {ridge.coef_}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.title('Actual vs Predicted Popularity')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
sns.lineplot(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
sns.regplot(x=y_test, y=y_pred, scatter=False, color='blue', line_kws={'label': 'Regression Line'})
plt.legend()
plt.show()

# Q8,  Q8


# Q9, Q9

data_mode_0 = data[data['mode'] == 0]
data_mode_1 = data[data['mode'] == 1]

print("Mode 0 Length" , len(data_mode_0));
print("Mode 1 Length" , len(data_mode_1));

plt.figure(figsize=(8, 6))
sns.histplot(data_mode_0['valence'], kde=True, color='blue', label='Mode 0')
plt.title('Distribution of Valence for Mode 0')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot distribution of 'valence' for mode 1
plt.figure(figsize=(8, 6))
sns.histplot(data_mode_1['valence'], kde=True, color='green', label='Mode 1')
plt.title('Distribution of Valence for Mode 1')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#features = ["duration", "danceability", "energy", "loudness", "speechiness",
           #"acousticness", "instrumentalness", "liveness", "valence", "tempo"]

accuracy_values_knn = []
sensitivity_values_knn = []
specificity_values_knn = []
roc_auc_values_knn = []

# Loop over each feature
for feature in features:
   # Select the current feature and the target variable
   dataQ9 = data[[feature, "mode"]].dropna()

   # Split the data into features (X) and target variable (y)
   X = dataQ9[[feature]]
   y = dataQ9["mode"].values

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train a k-NN model
   model_knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
   model_knn.fit(X_train, y_train)

   # Predict classes
   y_pred_knn = model_knn.predict(X_test)

   # Calculate confusion matrix
   cm_knn = confusion_matrix(y_test, y_pred_knn)

   # Calculate accuracy, sensitivity, and specificity
   accuracy_knn = (cm_knn[0, 0] + cm_knn[1, 1]) / np.sum(cm_knn)
   sensitivity_knn = cm_knn[1, 1] / (cm_knn[1, 0] + cm_knn[1, 1])
   specificity_knn = cm_knn[0, 0] / (cm_knn[0, 0] + cm_knn[0, 1])
   # Calculate ROC curve
   fpr, tpr, _ = roc_curve(y_test, model_knn.predict_proba(X_test)[:, 1])
   roc_auc = auc(fpr, tpr)

   # Append values to arrays
   accuracy_values_knn.append(accuracy_knn)
   sensitivity_values_knn.append(sensitivity_knn)
   specificity_values_knn.append(specificity_knn)
   roc_auc_values_knn.append(roc_auc)

   # Plot confusion matrix
   plt.figure()
   sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
   plt.xlabel('Predicted Class')
   plt.ylabel('Actual Class')
   plt.title(f"Confusion Matrix for {feature} (k-NN)\nAccuracy: {accuracy_knn:.3f}, Sensitivity: {sensitivity_knn:.3f}, Specificity: {specificity_knn:.3f}")
   plt.show()

   # Print metrics
   print(f"Feature: {feature}")
   print(f"Accuracy: {accuracy_knn:.3f}")
   print(f"Sensitivity: {sensitivity_knn:.3f}")
   print(f"Specificity: {specificity_knn:.3f}")
   print("\n")

# Print the best predictor and its accuracy for k-NN
best_predictor_knn = features[np.argmax(roc_auc_values_knn)]
best_auc_knn = max(roc_auc_values_knn)
print(f"Best Feature (k-NN): {best_predictor_knn}, Best AUC: {best_auc_knn:.3f}")

#Q10

label_encoder = LabelEncoder()
data['genre_label'] = label_encoder.fit_transform(data['track_genre'])

# Split data into features (X_pca) and target variable (y_genre)
X_pca = rotatedData
y_genre = data['genre_label']

# Split the data into training and testing sets
X_train_pca, X_test_pca, y_train_genre, y_test_genre = train_test_split(X_pca, y_genre, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(criterion='gini')  # You can adjust other parameters
decision_tree.fit(X_train_pca, y_train_genre)

# Predict probabilities for each class using predict_proba
y_prob_genre = decision_tree.predict_proba(X_test_pca)

# Evaluate the model
y_pred_genre = decision_tree.predict(X_test_pca)
accuracy_genre = accuracy_score(y_test_genre, y_pred_genre)
classification_rep_genre = classification_report(y_test_genre, y_pred_genre, target_names=label_encoder.classes_)

# Print the results
print(f"Genre Classification Accuracy: {accuracy_genre:.3f}")
print("\nGenre Classification Report:")
print(classification_rep_genre)

auc_roc_per_class = roc_auc_score(np.eye(len(label_encoder.classes_))[y_test_genre], y_prob_genre, multi_class='ovr')

# Plot the AUC-ROC curve
plt.figure(figsize=(10, 8))
max_auc_genre = None
max_auc_value = 0.0

for i, genre in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve((y_test_genre == i).astype(int), y_prob_genre[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{genre} (AUC = {roc_auc:.2f})')
    
    # Keep track of the genre with the maximum AUC
    if roc_auc > max_auc_value:
        max_auc_value = roc_auc
        max_auc_genre = genre

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve for Genre Classification')
plt.legend(loc='lower right')
plt.show()

# Print the genre with the highest AUC
print(f"The genre with the highest AUC is: {max_auc_genre} (AUC = {max_auc_value:.2f})")



# EXTRA CREDIT QUESTION 
# AN INDEPENDENT TTEST TO SEE IF SONGS WITH SHORTER NAMES ARE MORE POPULAR THEN SONGS WITH LONGER NAMES

data['name_length'] = data['track_name'].apply(len)

nameLenDescribe = data['name_length'].describe()
# 9 len 25% quartile, 23 75% quartile

# Define a threshold for short and long names (you can adjust this threshold)
short_name_threshold = 9
long_name_threshold = 23

# Create two groups: songs with short names and songs with long names
short_names = data[data['name_length'] <= short_name_threshold]['popularity']
long_names = data[data['name_length'] >= long_name_threshold]['popularity']

# Perform a two-sample t-test
t_statistic, p_value = ttest_ind(short_names, long_names)

# Calculate means for each group
mean_short_names = short_names.mean()
mean_long_names = long_names.mean()

# Interpret the findings
print(f'T-statistic: {t_statistic}, p-value: {p_value}')

if p_value < 0.05:
    if mean_short_names > mean_long_names:
        print("Songs with short names are more popular.")
    elif mean_short_names < mean_long_names:
        print("Songs with long names are more popular.")
    else:
        print("There is a significant difference in popularity, but no clear direction.")
else:
    print("There is no significant difference in popularity between songs with short and long names.")

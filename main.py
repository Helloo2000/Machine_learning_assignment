# Libraries
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy import percentile
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict 

# Load Data
buy = pd.read_csv('train.csv')

# Explore Data
print(buy.shape)
buy.head()
buy.dtypes

buy = buy.rename(columns = {'col1':'user_id', 'col2':'session_id','col3':'session_start_time', 'col4':'session_expiry_time',  'col5':'event_time',
 'col6':'event_time_zone', 'col7':'event_type', 'col8':'offer_decline_count', 'col9':'user_status', 'col10':'cart_quantity', 'col11':'cart_total',
 'col12':'last_offer_type', 'col13':'last_reward_value', 'col14':'last_spend_value', 'col15':'offer_display_count', 'col16':'user_screen_size',
 'col17':'offer_acceptance_state', 'col18':'converted'})

buy.describe()
buy.nunique()
# Get the unique values in 'converted'
unique_values = buy['converted'].unique()

# Print the unique values
print(unique_values)

# count the number of null values in each column of the DataFrame
null_counts = buy.isnull().sum(axis=0)

print(null_counts)

# find duplicates
dups = buy.duplicated()

# report if there are any duplicates
print(f"Are there any duplicates? {dups.any()}")
print(f"There are {dups.sum()} duplicate rows")

# create a heatmap of the correlation matrix between numerical columns
corr_matrix = buy.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

buy1 = buy.dropna()

all_box = buy1[['event_time_zone','offer_decline_count', 'cart_quantity', 'offer_display_count', 'last_spend_value', 'user_screen_size', 'last_reward_value', 'cart_total']]

# Create the figure and set the size
fig, ax = plt.subplots(figsize=(15, 6))

# Create the boxplot
ax.boxplot(all_box)

ax.set_xticklabels(['event_time_zone','offer_decline_count', 'cart_quantity', 'offer_display_count', 'last_spend_value', 'user_screen_size', 'last_reward_value', 'cart_total'])

# Normalize the data
normalized_df = all_box / all_box.mean()

# Create the figure and set the size
fig, ax = plt.subplots(figsize=(15, 6))

# Create the boxplot
ax.boxplot(normalized_df)

ax.set_xticklabels(['event_time_zone','offer_decline_count', 'cart_quantity', 'offer_display_count', 'last_spend_value', 'user_screen_size', 'last_reward_value', 'cart_total'])

# Print the variances of the normalized data
print(normalized_df.var())

# Baseline System
sample = buy1
sample = sample.drop(columns=['event_time_zone', 'offer_display_count', 'last_spend_value', 'user_screen_size', 'last_reward_value', 'cart_total'])

X = sample.iloc[:,:-1]
y = sample['converted']

# ordinal encoding of feature 
ordinal_encoder = OrdinalEncoder()
X = X.astype(str)
X = ordinal_encoder.fit_transform(X)

# encoding of target variable
y = y.astype(int)


# summarize the transformed data
print('Feature', X.shape)
print(X.dtype)
print(X[:5, :])
print('Target', y.shape)
print(y[:5])
y.nunique()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_labels = model.predict(X_test)
y_pred_scores = model.predict_proba(X_test)[:,1] 

y_pred_labels_train = model.predict(X_train)
y_pred_scores_train = model.predict_proba(X_train)[:, 1]

# Plot ROC curves for test set
fpr_test, tpr_test, thr_test = roc_curve(y_test, y_pred_scores)
auc_test = roc_auc_score(y_test, y_pred_scores)
plt.plot(fpr_test, tpr_test, '-b', label=f'Test AUC = {auc_test:.2f}')

# Plot ROC curves for train set
fpr_train, tpr_train, thr_train = roc_curve(y_train, y_pred_scores_train)
auc_train = roc_auc_score(y_train, y_pred_scores_train)
plt.plot(fpr_train, tpr_train, '-.r', label=f'Train AUC = {auc_train:.2f}')

plt.title('ROC Curves')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Improved System
subset = buy1.iloc[:,5:]

subset['last_value'] = subset['last_spend_value'] - subset['last_reward_value']

subset['offer_response_type'] = subset.apply(lambda x: 
    'Fixed Offer Accepted' if (x['offer_acceptance_state']=='ACCEPTED' and x['last_offer_type']=='F') 
    else 'Fixed Offer Declined' if (x['offer_acceptance_state']=='DECLINED' and x['last_offer_type']=='F') 
    else 'Variable Offer Accepted' if (x['offer_acceptance_state']=='ACCEPTED' and x['last_offer_type']!='F')
    else 'Variable Offer Declined', axis=1)

column_names = ['event_time_zone', 'event_type', 'offer_decline_count', 'user_status',
       'cart_quantity', 'cart_total', 'last_offer_type', 'last_reward_value',
       'last_spend_value', 'offer_display_count', 'user_screen_size',
       'offer_acceptance_state', 'last_value',
       'offer_response_type','converted']

# Use the reindex method to reorder the columns
subset = subset.reindex(columns=column_names)

subset = subset.drop(columns=['last_spend_value', 'last_reward_value', 'offer_acceptance_state', 'last_offer_type','offer_decline_count'])
subset.columns

X = subset.iloc[:,:-1]
y = subset['converted']

# ordinal encoding of feature 
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)

# encoding of target variable
y = y.astype(int)


# summarize the transformed data
print('Feature', X.shape)
print(X.dtype)
print(X[:5, :])
print('Target', y.shape)
print(y[:5])
y.nunique()

# define thresholds to check
thresholds = np.arange(0.0, 0.55, 0.05)

# apply transform with each threshold
results = list()
for t in thresholds:
	# define the transform
	transform = VarianceThreshold(threshold=t)
	# transform the input data
	X_sel = transform.fit_transform(X)
	# determine the number of input features
	n_features = X_sel.shape[1]
	print('>Threshold=%.2f, Features=%d' % (t, n_features))
	# store the result
	results.append(n_features)
# plot the threshold vs the number of selected features
plt.plot(thresholds, results)
plt.show()


# define the transform
transform = VarianceThreshold(threshold=0.2)
# transform the input data
X_sel = transform.fit_transform(X)
# determine the number of input features
n_features = X_sel.shape[1]
# get the Boolean mask of selected features
selected_features = transform.get_support()
# get the column names of the original feature matrix
feature_names = subset.columns
# print the selected features and their names
for feature, selected in zip(feature_names, selected_features):
    print(feature, selected)

X_selected = X[:, selected_features]


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_labels = model.predict(X_test)
y_pred_scores = model.predict_proba(X_test)[:,1] 

y_pred_labels_train = model.predict(X_train)
y_pred_scores_train = model.predict_proba(X_train)[:, 1]
# Plot ROC curves for test set
fpr_test, tpr_test, thr_test = roc_curve(y_test, y_pred_scores)
auc_test = roc_auc_score(y_test, y_pred_scores)
plt.plot(fpr_test, tpr_test, '-b', label=f'Test AUC = {auc_test:.2f}')

# Plot ROC curves for train set
fpr_train, tpr_train, thr_train = roc_curve(y_train, y_pred_scores_train)
auc_train = roc_auc_score(y_train, y_pred_scores_train)
plt.plot(fpr_train, tpr_train, '-.r', label=f'Train AUC = {auc_train:.2f}')

plt.title('ROC Curves')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("Creditcard_data.csv")

X = df.drop('Class', axis=1)
y = df['Class']

over_sampler = SMOTE(random_state=42)
X_resampled, y_resampled = over_sampler.fit_resample(X, y)

X_resampled = X_resampled.assign(Class=y_resampled)
df = X_resampled

M1 = []
M2 = []
M3 = []
M4 = []
M5 = []


# Support Vector Machine Classifier
def Model1(Sampling):
    X = Sampling.drop('Class', axis=1)
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_predicted = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    M1.append(accuracy * 100)


# Gradient Boosting Classifier
def Model2(Sampling):
    X = Sampling.drop('Class', axis=1)
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Create and train GradientBoostingClassifier
    gbc_classifier = GradientBoostingClassifier(random_state=42)
    gbc_classifier.fit(X_train, y_train)

    # Make predictions
    y_predicted = gbc_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predicted)

    M2.append(accuracy * 100)


# Logistic Regression
def Model3(Sampling):
    X = Sampling.drop('Class', axis=1)
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    logistic_reg_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_reg_model.fit(X_train, y_train)
    y_predicted = logistic_reg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    M3.append(accuracy * 100)


# Ada Boost Classifier
def Model4(Sampling):
    X = Sampling.drop('Class', axis=1)
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Create and train AdaBoostClassifier
    ada_classifier = AdaBoostClassifier(random_state=42)
    ada_classifier.fit(X_train, y_train)

    # Make predictions
    y_predicted = ada_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predicted)

    M4.append(accuracy * 100)


#  Naive Bayes Classifier
def Model5(Sampling):
    X = Sampling.drop('Class', axis=1)
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)
    y_predicted = naive_bayes_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    M5.append(accuracy * 100)


# Calculating sample size
sample_size = (1.960 ** 2 * 0.5 * (1 - 0.5)) / 0.05 ** 2
sample_size = int(sample_size)

# Performing simple random sampling
random_sample_indices = random.sample(list(df.index.values), sample_size)
Sampling1 = df.loc[random_sample_indices]

# performing systematic sampling
N = len(df)
sampling_interval = int(N / sample_size)
random_start = random.randint(0, sampling_interval - 1)
systematic_sample_indices = [random_start + i * sampling_interval for i in range(sample_size)]
Sampling2 = df.loc[systematic_sample_indices]

# performing cluster sampling
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df[['Amount']])
df['Cluster'] = kmeans.labels_
avg_cluster_size = 0.5
n_per_cluster = (1.960 ** 2 * 0.5 * (1 - 0.5)) / (0.05 / avg_cluster_size) ** 2
n_per_cluster = int(n_per_cluster)

if n_clusters * n_per_cluster > len(df):
    print("Error: Reduce the number of clusters")
    exit()

random_clusters = random.sample(list(df['Cluster'].unique()), n_clusters)

cluster_samples = []
for cluster in random_clusters:
    cluster_df = df[df['Cluster'] == cluster]
    n_to_sample = min(n_per_cluster, len(cluster_df))
    cluster_sample = cluster_df.sample(n_to_sample)
    cluster_samples.append(cluster_sample)

Sampling3 = pd.concat(cluster_samples)
Sampling3 = Sampling3.drop('Cluster', axis=1)
df = df.drop('Cluster', axis=1)


# performing stratified sampling
def calculate_sample_size(N):
    return int((1.960 ** 2 * 0.5 * (1 - 0.5)) / ((0.05 / np.sqrt(N)) ** 2))


def stratified_sample(df, column, sample_size, random_state=42):
    stratum_sampled_data = pd.DataFrame(columns=df.columns)
    for stratum_value, stratum_df in df.groupby(column):
        stratum_size = len(stratum_df)
        stratum_sample_size = calculate_sample_size(0.5)

        stratum_sample_size = min(stratum_sample_size, stratum_size)

        stratum_sample = stratum_df.sample(n=stratum_sample_size, random_state=random_state)

        stratum_sampled_data = pd.concat([stratum_sampled_data, stratum_sample])

    return stratum_sampled_data


Sampling4 = stratified_sample(df, 'Class', sample_size, random_state=42)
Sampling4['Class'] = Sampling4['Class'].astype(int)

# performing boostrap sampling

sample_size = (1.960 ** 2 * 0.5 * (1 - 0.5)) / 0.05 ** 2
sample_size = int(sample_size)

Sampling5 = df.sample(n=sample_size, replace=True)

list_of_samples = [Sampling1, Sampling2, Sampling3, Sampling4, Sampling5]
for i in range(0, 5):
    Model1(list_of_samples[i])
    Model2(list_of_samples[i])
    Model3(list_of_samples[i])
    Model4(list_of_samples[i])
    Model5(list_of_samples[i])

data = [M1, M2, M3, M4, M5]
columns = ['Simple Random ', 'Systematic', 'Cluster', 'Stratified ', 'Bootstrap']
index = ['SVM', 'Gradient Boosting', 'Logistic Regression', 'Ada Boost Classifier', 'Naive Bayes']
Final = pd.DataFrame(data, columns=columns, index=index)

maxValue = -1
final_model = None
final_sample = None
for model in Final.index:
    for sample in Final.columns:
        value = Final.loc[model][sample]
        if value > maxValue:
            maxValue = value
            final_model = model
            final_sample = sample

print(Final)

print(f"The maximum value {maxValue} is given by {final_model} Model and  {final_sample} Sampling")
Final.to_csv("Submission.csv", index=False)

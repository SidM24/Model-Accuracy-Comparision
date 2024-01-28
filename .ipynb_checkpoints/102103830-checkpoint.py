import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 
import random 
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("Creditcard_data.csv")



X = df.drop('Class', axis=1)  
y = df['Class']

oversampler = SMOTE(random_state=42)  
X_resampled, y_resampled = oversampler.fit_resample(X, y)



X_resampled = X_resampled.assign(Class = y_resampled)
df = X_resampled

M1 = []
M2 =[]
M3 = []
M4 =[]
M5 = []
def Model1(Sampling):
    X = Sampling.drop('Class', axis=1)  
    y = Sampling['Class']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  
    svm_classifier = SVC(kernel='linear', random_state=42)

    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    M1.append(accuracy*100)
def Model2(Sampling):
    X = Sampling.drop('Class', axis=1)  
    y = Sampling['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    
    logistic_reg_model = LogisticRegression(random_state=42,max_iter=1000)

   
    logistic_reg_model.fit(X_train, y_train)

    
    y_pred = logistic_reg_model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    
    M2.append(accuracy*100)
def Model3(Sampling):
    X = Sampling.drop('Class', axis=1)  
    y = Sampling['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  
    tree_classifier = DecisionTreeClassifier(random_state=42)

    tree_classifier.fit(X_train, y_train)

    y_pred = tree_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    M3.append(accuracy*100)
def Model4(Sampling):
    X = Sampling.drop('Class', axis=1)  
    y = Sampling['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    
    M4.append(accuracy*100)

def Model5(Sampling):
    X = Sampling.drop('Class', axis=1)  
    y = Sampling['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  
    naive_bayes_classifier = GaussianNB()

    naive_bayes_classifier.fit(X_train, y_train)

    y_pred = naive_bayes_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    M5.append(accuracy*100)

#Performing simple random sampling
    
sample_size = (1.960**2 * 0.5 * (1-0.5)) / 0.05**2 
sample_size = int(sample_size)

random_sample_indices = random.sample(list(df.index.values), sample_size)
Sampling1 = df.loc[random_sample_indices]
Model1(Sampling1)
Model2(Sampling1)
Model3(Sampling1)
Model4(Sampling1)
Model5(Sampling1)

#performing systematic sampling

N = len(df)  
sampling_interval = int(N / sample_size)

random_start = random.randint(0, sampling_interval - 1)

systematic_sample_indices = [random_start + i * sampling_interval for i in range(sample_size)]


Sampling2 = df.loc[systematic_sample_indices]
Model1(Sampling2)
Model2(Sampling2)
Model3(Sampling2)
Model4(Sampling2)
Model5(Sampling2)


#performing cluster sampling
n_clusters = 6


kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df[['Amount']])
df['Cluster'] = kmeans.labels_

avg_cluster_size = 0.5


n_per_cluster = (1.960**2 * 0.5 * (1-0.5)) / (0.05/avg_cluster_size)**2 
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
Sampling3 = Sampling3.drop('Cluster',axis=1)
df = df.drop('Cluster',axis=1)

Model1(Sampling3)
Model2(Sampling3)
Model3(Sampling3)
Model4(Sampling3)
Model5(Sampling3)

#performing stratified sampling
def calculate_sample_size(N):
    return int((1.960**2 * 0.5 * (1 - 0.5)) / ((0.05 / np.sqrt(N))**2))

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

Model1(Sampling4)
Model2(Sampling4)
Model3(Sampling4)
Model4(Sampling4)
Model5(Sampling4)






#performing boostrap sampling

sample_size = (1.960**2 * 0.5 * (1-0.5)) / 0.05**2
sample_size = int(sample_size)


Sampling5 = df.sample(n=sample_size, replace=True)
Model1(Sampling5)
Model2(Sampling5)
Model3(Sampling5)
Model4(Sampling5)
Model5(Sampling5)

data = [M1, M2, M3, M4, M5]
columns = ['Simple Random ', 'Systematic', 'Cluster', 'Stratified ', 'Bootstrap']
index = ['SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes']

Final = pd.DataFrame(data, columns=columns, index=index)

maxValue = -1
final_model = None
final_sample = None
for model in Final.index:
    for sample in Final.columns:
        value = Final.loc[model][sample]
        if value>maxValue:
            maxValue=value
            final_model = model
            final_sample = sample

print(Final)




print(f"The maximum value {maxValue} is found in  {final_model} Model and  {final_sample} Sampling")
Final.to_csv("Submission.csv",index=False)






# Data Science
# Sampling
## Problem: Balancing the imbalanced
The Credit card dataset presents a significant class imbalance, featuring merely 9 instances labeled as Class '1' and a substantial majority of over 700 instances labeled as Class '0'. Consequently, if the model were to predict all entries as '0' or 'not fraud', the accuracy would misleadingly appear high, surpassing 95%.

To address this imbalance, a prudent strategy involves implementing either Over Sampling or Under Sampling techniques. Given the exceedingly low count of Class '1', the preferred approach is to employ Oversampling, specifically utilizing the **SMOTE (Synthetic Minority Oversampling Technique) algorithm**. This method involves generating synthetic instances of the minority class to create a more balanced dataset, thereby enhancing the model's ability to discern and generalize patterns associated with the minority class.

## Sampling:
I have used 5 techniques for sampling which are:
1. **Simple Random Sampling**
2. **Systematic Sampling**
3. **Cluster Sampling**
4. **Stratified Sampling**
5. **Bootstrap**

## Classification Models:
This is a binary classification task for which I have used 5 classification models:-
1. **Support Vector Machine**
2. **Gradient Boosting**
3. **Logistic Regression**
4. **Ada Boost Classifier**
5. **Naive Bayes**

This is my Final result after applying all models on all types of sampling

| Classifiers          | Simple Random | Systematic  | Cluster     | Stratified  | Bootstrap   |
|----------------------|---------------|-------------|-------------|-------------|-------------|
| SVM                  | 91.66666667   | 88.02083333 | 91.66666667 | 89.58333333 | 91.14583333 |
| Gradient Boosting    | 95.3125       | 92.1875     | 90          | 94.79166667 | 92.70833333 |
| Logistic Regression  | 94.27083333   | 89.0625     | 93.33333333 | 92.1875     | 91.66666667 |
| Ada Boost Classifier | 95.83333333   | 95.83333333 | 94.16666667 | 94.79166667 | 91.14583333 |
| Naive Bayes          | 73.95833333   | 76.04166667 | 94.16666667 | 83.33333333 | 84.89583333 |
### The maximum value 95.83333333333334 is given by Ada Boost Classifier Model and  Simple Random  Sampling









# Titanic Survival Prediction
This project will focus on the Titanic dataset available on <a href="https://www.kaggle.com/competitions/titanic">Kaggle</a>. The objective of the project is to predict passenger survival based on their personal attributes. Detailed information about each attribute can be found on the same page where the dataset can be downloaded.

## Feature Classification
Below is the Python code used to extract the training and test data from Kaggle and store them in the variables `data_train` and `data_test`, respectively:

```python
import pandas as pd
data_test = pd.read_csv("test.csv")
data_train = pd.read_csv("train.csv")
```
We can then utilize the following code to visualize the information of the `data_train` variable, which is represented in the `Table 1`.

```python
data_train.info()
```
### Table 1: Data Description

| Column      | Non-Null Count | Dtype   |
|-------------|----------------|---------|
| PassengerId | 891            | int64   |
| Survived    | 891            | int64   |
| Pclass      | 891            | int64   |
| Name        | 891            | object  |
| Sex         | 891            | object  |
| Age         | 714            | float64 |
| SibSp       | 891            | int64   |
| Parch       | 891            | int64   |
| Ticket      | 891            | object  |
| Fare        | 891            | float64 |
| Cabin       | 204            | object  |
| Embarked    | 889            | object  |


In `Table 1`, it can be observed how the columns are of different data types. The following columns are of numeric type: PassengerId, Survived, Pclass, Age, SibSp, Parch, and Fare. On the other hand, the following columns are of object type: Name, Sex, Ticket, Cabin, and Embarked.

Regarding missing values in the dataset, it has been noted that the following columns have missing values: Age, Cabin, and Embarked. The Age column has 177 missing values, the Cabin column has 687, and the Embarked column has 2. Due to the Cabin column having a large number of missing values (approximately 77.1% of its values), it has been decided not to consider it in further analysis.

To handle missing values in the Age column, the 'median' strategy has been used through the SimpleImputer function, which fills missing values with the median. On the other hand, for missing values in the Embarked column, the 'most frequent' strategy has been used in the same function, assigning the most frequent value in the column to the missing values.

To analyze the data type of the columns Name, Sex, Ticket, and Embarked, the following code can be used, which produces `Table 2` showing the number of unique values in each column of the dataset.

```python
data_train[["Name", "Sex", "Ticket","Embarked"]].nunique()
```
### Table 2: Number of Unique Values per Column

| Column   | Unique Values |
|----------|---------------|
| Name     | 891           |
| Sex      | 2             |
| Ticket   | 681           |
| Embarked | 3             |

From `Table 2`, it can be observed that all passengers have a different Name, and some passengers have the same ticket number, but they are different for each person in principle. However, these columns do not provide relevant information for model training, as they contain a large number of unique values. On the other hand, the Sex and Embarked columns are categorical and will be used in model training.

To see the different values ​​taken by the categorical columns Sex and Embarked, the following code can be used, which shows that the Sex column has the categories ['male' 'female'] and the Embarked column has the categories ['S' 'C' 'Q' nan'].

```python
data_train['Embarked'].unique()
data_train['Sex'].unique()
```
Therefore, for the modeling process, the PassengerId, Survived, Pclass, Age, SibSp, Parch, and Fare columns will be considered as numerical attributes. While the Sex and Embarked columns will be considered as categorical attributes.

## Relevance of Variables for the Problem

For numerical attributes, the following code can be used to show the correlation between the different variables and passenger survival, resulting in `Table 3`.

```python
corr_matrix = data_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
```
### Table 3: Correlation between variables and survival

| Variable    | Correlation with Survived |
|-------------|---------------------------|
| Survived    | 1.000000                  |
| Fare        | 0.257307                  |
| Parch       | 0.081629                  |
| PassengerId | -0.005007                 |
| SibSp       | -0.035322                 |
| Age         | -0.077221                 |
| Pclass      | -0.338481                 |

From `Table 3`, it can be concluded that the variable Survived is positively correlated with the Fare variable, suggesting that people who paid a higher fare have a higher probability of survival. Additionally, the Parch variable has a weak positive correlation with the Survived variable, indicating that people traveling with parents or children had a slightly higher chance of survival.

The PassengerId variable does not appear to have a significant correlation with survival, which makes sense because it is only a unique identifier for each passenger. Therefore, it will not be used in model training.

On the other hand, the Age, Pclass, and SibSp variables are negatively correlated with the Survived variable, suggesting that older people, those traveling in lower classes, and those with more siblings/spouses aboard had a lower probability of survival.

To analyze the relationship of categorical attributes with survival, the following code can be used to generate a stacked bar chart showing the number of survivors and fatalities for each category. The results are shown in `Figure 1`.

```python
import matplotlib.pyplot as plt

# Create a dataframe with the number of survivors and fatalities for each gender
gender_survival = pd.crosstab(data_train['Sex'], data_train['Survived'])
# Create a dataframe with the number of survivors and fatalities for each port of embarkation
embarked_survival = pd.crosstab(data_train['Embarked'], data_train['Survived'])

# Create a bar chart
gender_survival.plot(kind='bar', stacked=True)

# Add labels
plt.xlabel('Gender')
plt.ylabel('Quantity')
plt.title('Relationship between gender and survival')
plt.legend(['Fatalities', 'Survivors'])

# Create a bar chart
embarked_survival.plot(kind='bar', stacked=True)

# Add labels
plt.xlabel('Embarkation Port')
plt.ylabel('Quantity')
plt.title('Relationship between embarkation port and survival')
plt.legend(['Fatalities', 'Survivors'])

plt.show()
```
### Figure 1: Relationship between Gender, Embarkation Port and Survival

<table>
  <tr>
    <td><img src="gender.png" alt="Figure 1"></td>
    <td><img src="embarkation.png" alt="Figure 2"></td>
  </tr>
</table>

In the Embarked graph, it can be observed that the majority of passengers embarked at port S, and passengers who embarked at port C have the highest survival rate, while those who embarked at port S have the lowest survival rate. In the Sex graph, it can be seen that females have a much higher survival rate than males. This suggests that both the embarkation port and gender may be important factors in determining the probability of survival.

## Performance Comparison between Pipelines on Training Data
Three pipelines, namely `Pipeline_1`, `Pipeline_2`, and `Pipeline_3`, were created and evaluated based on their performance on the training data.

### Pipeline_1:
The `Pipeline_1` was constructed by excluding the attributes Cabin, Name, Ticket, and Survived, where the latter serves as the target variable of the model. The following code snippet illustrates the pipeline creation:

```python
# Saved attribute names to be used according to their type
num_attribs = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare','Pclass']
cat_attribs = ['Sex', 'Embarked']

# Created a pipeline for numerical attributes and another for categorical attributes
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

# Preprocessed the data, applying each pipeline to the corresponding attributes
pipeline_1 = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)])
data_train_prepared = pipeline_1.fit_transform(data_train)
```
### Pipeline_2:
The `Pipeline_2` was built by excluding all attributes except for the gender (Sex), as demonstrated in the code below:

```python
pipeline_2 = ColumnTransformer([("cat", cat_pipeline, ["Sex"]),])
data_train_prepared_2 = pipeline_2.fit_transform(data_train)
```
### Pipeline_3:
In the `Pipeline_3`, two changes were made in the attribute selection. Firstly, the PassengerId attribute was discarded due to its low correlation with the target variable Survived. Secondly, the Pclass attribute, which has only three different values, was included in the set of categorical attributes. The following code was used to implement these transformations:

```python
num_attribs_3 = ['Age', 'SibSp', 'Parch', 'Fare']
cat_attribs_3 = ['Pclass','Sex', 'Embarked']
pipeline_3 = ColumnTransformer([("num", num_pipeline, num_attribs_3), ("cat", cat_pipeline, cat_attribs_3)])
data_train_prepared_3 = pipeline_3.fit_transform(data_train)
```
A Logistic Regression model with default parameters was trained and evaluated using 5-fold cross-validation. The performance results are displayed in `Table 4`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_reg = LogisticRegression().fit(data_train_prepared, y_target)
log_reg_2 = LogisticRegression().fit(data_train_prepared_2, y_target)
log_reg_3 = LogisticRegression().fit(data_train_prepared_3, y_target)

scores = cross_val_score(log_reg, data_train_prepared, y_target, cv=5)
scores_2 = cross_val_score(log_reg_2, data_train_prepared_2, y_target, cv=5)
scores_3 = cross_val_score(log_reg_3, data_train_prepared_3, y_target, cv=5)
```

### Table 4: Comparison of cross-validation scores for the three pipelines

| Pipeline      | Cross-validation scores                                          | Average   | Standard Deviation |
|---------------|------------------------------------------------------------------|-----------|--------------------|
| Pipeline\_1   | [0.782123, 0.786517, 0.780899, 0.769663, 0.814607]              | 0.786762  | 0.014991           |
| Pipeline\_2   | [0.804469, 0.803371, 0.786517, 0.752809, 0.786517]              | 0.786737  | 0.018667           |
| Pipeline\_3   | [0.787709, 0.786517, 0.786517, 0.769663, 0.831461]              | 0.792373  | 0.020659           |

From `Table 4`, it can be concluded that `Pipeline_3` exhibits the best performance, with an average of 0.792 and a standard deviation of 0.021.
To improve the performance of `Pipeline_3`, a RandomForestClassifier is employed, and GridSearchCV is used to find the best parameters.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Parameters to be evaluated
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
rfc = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

# Train and fit the model
grid_search.fit(data_train_prepared_3, y_target)
```
Using the best parameters, a new `Pipeline_4` is created, and its performance is added to `Table 4`, resulting in `Table 5`.

```python
classifier = RandomForestClassifier(max_depth=None, max_features='log2', 
min_samples_leaf=2, min_samples_split=2, n_estimators=50)
pipeline_4 = make_pipeline(pipeline_3, classifier)
scores_4 = cross_val_score(pipeline_4, data_train, y_target, cv=5)
```

### Table 5: Comparison of cross-validation scores for the four pipelines

| Pipeline     | Cross-validation scores                               | Average   | Standard Deviation |
|--------------|-------------------------------------------------------|-----------|--------------------|
| Pipeline\_1  | [0.782123, 0.786517, 0.780899, 0.769663, 0.814607]   | 0.786762  | 0.014991           |
| Pipeline\_2  | [0.804469, 0.803371, 0.786517, 0.752809, 0.786517]   | 0.786737  | 0.018667           |
| Pipeline\_3  | [0.787709, 0.786517, 0.786517, 0.769663, 0.831461]   | 0.792373  | 0.020659           |
| Pipeline\_4  | [0.793296, 0.803371, 0.865169, 0.814607, 0.831461]   | 0.821581  | 0.025210           |

It can be observed that pipeline_4 performed better compared to the other pipelines. Therefore, it will be saved as the final model and evaluated with the test data.

## Final Model

The `Figure 2` shows the result obtained when evaluating pipeline_4 with the test set available on the Kaggle platform.

### Figure 2: Score obtained on the Kaggle platform

<table>
  <tr>
    <td><img src="score.png" alt="Figure 1"></td>
  </tr>
</table>

It can be inferred that the model generalizes well, as the value obtained in the test set is similar to the value obtained in the training set.

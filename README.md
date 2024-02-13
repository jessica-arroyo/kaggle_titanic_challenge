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

To analyze the relationship of categorical attributes with survival, the following code can be used to generate a stacked bar chart showing the number of survivors and fatalities for each category. The results are shown in Figure \ref{fig:figura_general}.

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
![Description of the Image](images/gender.png)
![Description of the Image](images/embarkation.png)


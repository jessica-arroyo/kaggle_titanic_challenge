# Titanic Survival Prediction
This project will focus on the Titanic dataset available on <a href="https://www.kaggle.com/competitions/titanic">Kaggle</a>. The objective of the project is to predict passenger survival based on their personal attributes. Detailed information about each attribute can be found on the same page where the dataset can be downloaded.

## Feature Classification
Below is the Python code used to extract the training and test data from Kaggle and store them in the variables `data_train` and `data_test`, respectively:

```python
import pandas as pd
data_test = pd.read_csv("test.csv")
data_train = pd.read_csv("train.csv")
```
We can then utilize the following code to visualize the information of the `data_train` variable, which is represented in the [Table 1](#tabla).

```python
data_train.info()
```
### Table 1: Data Description

<table id="table1">
    <thead>
        <tr>
            <th>Column</th>
            <th>Non-Null Count</th>
            <th>Dtype</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>PassengerId</td>
            <td>891</td>
            <td>int64</td>
        </tr>
        <tr>
            <td>Survived</td>
            <td>891</td>
            <td>int64</td>
        </tr>
        <tr>
            <td>Pclass</td>
            <td>891</td>
            <td>int64</td>
        </tr>
        <tr>
            <td>Name</td>
            <td>891</td>
            <td>object</td>
        </tr>
        <tr>
            <td>Sex</td>
            <td>891</td>
            <td>object</td>
        </tr>
        <tr>
            <td>Age</td>
            <td>714</td>
            <td>float64</td>
        </tr>
        <tr>
            <td>SibSp</td>
            <td>891</td>
            <td>int64</td>
        </tr>
        <tr>
            <td>Parch</td>
            <td>891</td>
            <td>int64</td>
        </tr>
        <tr>
            <td>Ticket</td>
            <td>891</td>
            <td>object</td>
        </tr>
        <tr>
            <td>Fare</td>
            <td>891</td>
            <td>float64</td>
        </tr>
        <tr>
            <td>Cabin</td>
            <td>204</td>
            <td>object</td>
        </tr>
        <tr>
            <td>Embarked</td>
            <td>889</td>
            <td>object</td>
        </tr>
    </tbody>
</table>

In [Table 1](#table1), it can be observed how the columns are of different data types. The following columns are of numeric type: PassengerId, Survived, Pclass, Age, SibSp, Parch, and Fare. On the other hand, the following columns are of object type: Name, Sex, Ticket, Cabin, and Embarked.

Regarding missing values in the dataset, it has been noted that the following columns have missing values: Age, Cabin, and Embarked. The Age column has 177 missing values, the Cabin column has 687, and the Embarked column has 2. Due to the Cabin column having a large number of missing values (approximately 77.1% of its values), it has been decided not to consider it in further analysis.

To handle missing values in the Age column, the 'median' strategy has been used through the SimpleImputer function, which fills missing values with the median. On the other hand, for missing values in the Embarked column, the 'most frequent' strategy has been used in the same function, assigning the most frequent value in the column to the missing values.

To analyze the data type of the columns Name, Sex, Ticket, and Embarked, the following code can be used, which produces [Table 2](#table2) showing the number of unique values in each column of the dataset.

```python
data_train[["Name", "Sex", "Ticket","Embarked"]].nunique()
```



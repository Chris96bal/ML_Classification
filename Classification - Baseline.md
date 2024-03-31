## In this section, we will start by performing different Classification models to create a baseline. Then we will proceed by performing Hyperparameter Tuning and employing Ensemble methods and try creating a higher accuracy model.

### Firstly, importing all necessary libraries is performed


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")
```

### Reading and storing the training dataset


```python
train = pd.read_csv("CS98XClassificationTrain.csv")
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>title</th>
      <th>artist</th>
      <th>year</th>
      <th>bpm</th>
      <th>nrgy</th>
      <th>dnce</th>
      <th>dB</th>
      <th>live</th>
      <th>val</th>
      <th>dur</th>
      <th>acous</th>
      <th>spch</th>
      <th>pop</th>
      <th>top genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>My Happiness</td>
      <td>Connie Francis</td>
      <td>1996</td>
      <td>107</td>
      <td>31</td>
      <td>45</td>
      <td>-8</td>
      <td>13</td>
      <td>28</td>
      <td>150</td>
      <td>75</td>
      <td>3</td>
      <td>44</td>
      <td>adult standards</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Unchained Melody</td>
      <td>The Teddy Bears</td>
      <td>2011</td>
      <td>114</td>
      <td>44</td>
      <td>53</td>
      <td>-8</td>
      <td>13</td>
      <td>47</td>
      <td>139</td>
      <td>49</td>
      <td>3</td>
      <td>37</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>How Deep Is Your Love</td>
      <td>Bee Gees</td>
      <td>1979</td>
      <td>105</td>
      <td>36</td>
      <td>63</td>
      <td>-9</td>
      <td>13</td>
      <td>67</td>
      <td>245</td>
      <td>11</td>
      <td>3</td>
      <td>77</td>
      <td>adult standards</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Woman in Love</td>
      <td>Barbra Streisand</td>
      <td>1980</td>
      <td>170</td>
      <td>28</td>
      <td>47</td>
      <td>-16</td>
      <td>13</td>
      <td>33</td>
      <td>232</td>
      <td>25</td>
      <td>3</td>
      <td>67</td>
      <td>adult standards</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Goodbye Yellow Brick Road - Remastered 2014</td>
      <td>Elton John</td>
      <td>1973</td>
      <td>121</td>
      <td>47</td>
      <td>56</td>
      <td>-8</td>
      <td>15</td>
      <td>40</td>
      <td>193</td>
      <td>45</td>
      <td>3</td>
      <td>63</td>
      <td>glam rock</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>448</th>
      <td>449</td>
      <td>But Not For Me</td>
      <td>Ella Fitzgerald</td>
      <td>1959</td>
      <td>80</td>
      <td>22</td>
      <td>18</td>
      <td>-17</td>
      <td>10</td>
      <td>16</td>
      <td>214</td>
      <td>92</td>
      <td>4</td>
      <td>45</td>
      <td>adult standards</td>
    </tr>
    <tr>
      <th>449</th>
      <td>450</td>
      <td>Surf City</td>
      <td>Jan &amp; Dean</td>
      <td>2010</td>
      <td>148</td>
      <td>81</td>
      <td>53</td>
      <td>-13</td>
      <td>23</td>
      <td>96</td>
      <td>147</td>
      <td>50</td>
      <td>3</td>
      <td>50</td>
      <td>brill building pop</td>
    </tr>
    <tr>
      <th>450</th>
      <td>451</td>
      <td>Dilemma</td>
      <td>Nelly</td>
      <td>2002</td>
      <td>168</td>
      <td>55</td>
      <td>73</td>
      <td>-8</td>
      <td>20</td>
      <td>61</td>
      <td>289</td>
      <td>23</td>
      <td>14</td>
      <td>77</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>451</th>
      <td>452</td>
      <td>It's Gonna Be Me</td>
      <td>*NSYNC</td>
      <td>2000</td>
      <td>165</td>
      <td>87</td>
      <td>64</td>
      <td>-5</td>
      <td>6</td>
      <td>88</td>
      <td>191</td>
      <td>5</td>
      <td>8</td>
      <td>62</td>
      <td>boy band</td>
    </tr>
    <tr>
      <th>452</th>
      <td>453</td>
      <td>In The Army Now</td>
      <td>Status Quo</td>
      <td>2002</td>
      <td>105</td>
      <td>73</td>
      <td>68</td>
      <td>-8</td>
      <td>14</td>
      <td>94</td>
      <td>281</td>
      <td>11</td>
      <td>2</td>
      <td>59</td>
      <td>album rock</td>
    </tr>
  </tbody>
</table>
<p>453 rows × 15 columns</p>
</div>



### In this part, we will keep only the numerical columns as independent variables excluding the "Id" column (Columns "title" and "artist" will be removed).


```python
train = train.drop(['Id','title','artist'],axis=1)
```

#### Id is useless because is different for every row and serves solely as a row identificator. Furthermore, the columns title and
#### artist will not yet get examined as we only want to implement some simple classification models which will serve as a baseline.

### Checking for null values


```python
train.isna().sum().sort_values(ascending=False)
```




    top genre    15
    year          0
    bpm           0
    nrgy          0
    dnce          0
    dB            0
    live          0
    val           0
    dur           0
    acous         0
    spch          0
    pop           0
    dtype: int64



#### There are 15 null values on the dataset, which will be removed to offer more clean and credible data.


```python
train=train.dropna()
```

#### We will also reset the index to know exactly how many rows the dataset is consisted of.


```python
train = train.reset_index()
train = train.drop(["index"], axis = 1)
```

#### We will identify what kind of values we have in our dataset


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 438 entries, 0 to 437
    Data columns (total 12 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   year       438 non-null    int64 
     1   bpm        438 non-null    int64 
     2   nrgy       438 non-null    int64 
     3   dnce       438 non-null    int64 
     4   dB         438 non-null    int64 
     5   live       438 non-null    int64 
     6   val        438 non-null    int64 
     7   dur        438 non-null    int64 
     8   acous      438 non-null    int64 
     9   spch       438 non-null    int64 
     10  pop        438 non-null    int64 
     11  top genre  438 non-null    object
    dtypes: int64(11), object(1)
    memory usage: 41.2+ KB
    

#### It is easy to see that every column contains integer values, with the exception of the "top genre" column which is consisted
#### of categorical variables. Finally, the remaining rows, without Null values, are 438.

#### ========================================================================================================

#### Now, we will use the describe() function to gain a deeper understanding of each numerical column of the dataset


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>bpm</th>
      <th>nrgy</th>
      <th>dnce</th>
      <th>dB</th>
      <th>live</th>
      <th>val</th>
      <th>dur</th>
      <th>acous</th>
      <th>spch</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
      <td>438.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1990.881279</td>
      <td>118.326484</td>
      <td>60.504566</td>
      <td>59.780822</td>
      <td>-8.787671</td>
      <td>17.605023</td>
      <td>59.625571</td>
      <td>228.267123</td>
      <td>32.191781</td>
      <td>5.671233</td>
      <td>61.557078</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.697047</td>
      <td>25.175735</td>
      <td>22.089660</td>
      <td>15.404757</td>
      <td>3.591005</td>
      <td>13.807492</td>
      <td>24.480160</td>
      <td>63.426812</td>
      <td>29.279912</td>
      <td>5.571392</td>
      <td>12.759353</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1948.000000</td>
      <td>62.000000</td>
      <td>7.000000</td>
      <td>18.000000</td>
      <td>-24.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>98.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1976.000000</td>
      <td>100.000000</td>
      <td>44.000000</td>
      <td>50.000000</td>
      <td>-11.000000</td>
      <td>9.000000</td>
      <td>42.250000</td>
      <td>184.500000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1993.000000</td>
      <td>120.000000</td>
      <td>64.000000</td>
      <td>62.000000</td>
      <td>-8.000000</td>
      <td>13.000000</td>
      <td>61.000000</td>
      <td>224.000000</td>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>64.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2006.000000</td>
      <td>133.000000</td>
      <td>78.000000</td>
      <td>70.750000</td>
      <td>-6.000000</td>
      <td>23.000000</td>
      <td>80.000000</td>
      <td>264.000000</td>
      <td>57.000000</td>
      <td>6.000000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019.000000</td>
      <td>199.000000</td>
      <td>100.000000</td>
      <td>96.000000</td>
      <td>-1.000000</td>
      <td>93.000000</td>
      <td>99.000000</td>
      <td>511.000000</td>
      <td>99.000000</td>
      <td>47.000000</td>
      <td>84.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### After describing the data, it is obvious that some variables such as dB, live, dur, acous and spch do not follow a normal
#### distribution and are skewed and may create bias if applied to a model. As a result we will scale the necessary columns to 
#### avoid such potential issues. To gain a better understanding about the skewed variables we will use visuals.

#### In this analysis the objective is to make a model which will be able to classify songs to their respective genre. So, the
#### next bar - graph reveals how many different genres exist within the dataset:


```python
train["top genre"].value_counts().plot.bar(color="maroon",grid=True, figsize=(22,8))
plt.xlabel("Type of Genre",fontsize = "xx-large")
plt.ylabel("Total number of songs",fontsize = "xx-large")
plt.title("Total number of song of each Genre",fontsize = "xx-large")
plt.show()
```


    
![png](output_22_0.png)
    



```python
train["top genre"].value_counts()
```




    adult standards       68
    album rock            66
    dance pop             61
    brill building pop    16
    glam rock             16
                          ..
    bow pop                1
    australian rock        1
    boogaloo               1
    british comedy         1
    alternative rock       1
    Name: top genre, Length: 86, dtype: int64



#### We see that there are 86 different genres which are too many to build an accurate classification model. An additional issue is
#### that not every genre is consisted with the same number of songs. For example there are several genres which are consisted of only
#### one or two songs. As a result, this class imblance will present difficulties for any model to train data and classify them to these genres.

### The following subplots will provide a better idea of each column's distribution


```python
fig,ax = plt.subplots(3,4,figsize=(20,10))

sns.distplot(train['year'],ax=ax[0,0])
sns.distplot(train['bpm'],ax=ax[0,1])
sns.distplot(train['nrgy'],ax=ax[0,2])
sns.distplot(train['dnce'],ax=ax[0,3])
#sns.distplot(df['duration_min'],ax=ax[1,1])
sns.distplot(train['dB'],ax=ax[1,0])
sns.distplot(train['live'],ax=ax[1,1])
sns.distplot(train['val'],ax=ax[1,2])
sns.distplot(train['dur'],ax=ax[1,3])
sns.distplot(train['acous'],ax=ax[2,0])
sns.distplot(train['spch'],ax=ax[2,1])
sns.distplot(train['pop'],ax=ax[2,2])

plt.suptitle("Distribution of each independent variable",fontsize = "xx-large")
plt.show()
```


    
![png](output_26_0.png)
    


#### Like said above, variables such as dB, live, dur, acous and spch do noth follow a normal distribution and are skewed.

#### Another useful thing is to check the variables for potential multicollinearity between them. If multicollinearity exists between two variables, it is better to remove one of them. Otherwise the performance of the classification model will drop.
#### Another way to deal with multicollinearity is to use PCA (Principal Component Analysis) which takes advantage of multicollinearity and combines the highly correlated variables into a set of uncorrelated variables.


```python
plt.figure(figsize=(10,8)) 
sns.heatmap(train.corr(), annot=True)
plt.title("Correlation between independent variables", fontsize = "xx-large")
plt.show()
```


    
![png](output_29_0.png)
    


#### From the heatmap, it is clear that there are no signs of multicollinearity beween the variables. The variables "nrgy" and "dB" are slightly correlated, but we will not remove any of them in this part of analysis.

### Data Scaling


```python
scaler = StandardScaler()
train[["acous", "spch", "live"]] = scaler.fit_transform(train[["acous", "spch", "live"]])
```

#### Only the most skewed data was scaled.

### Model Building


```python
Y = train.loc[:, 'top genre']
X = train.drop(['top genre'], axis=1)
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)
```

#### Firstly we separated the data to values and labels. The values are consisted of the columns year, bpm, nrgy, dnce, dB, live
#### val, dur, acous, spch, pop and the labels which is the "top genre" column. Lastly, data are splitted to training-test sets.

### Firstly, we will try several classifiers to compare accuracy.


```python
tree_clf = DecisionTreeClassifier()
log_clf = LogisticRegression()
ovr_clf = OneVsRestClassifier(LogisticRegression())
ovr2_clf = OneVsRestClassifier(SVC())
rnd_clf = RandomForestClassifier()
kn_clf = KNeighborsClassifier()
svm_clf = SVC(kernel='poly', degree=2, C=1)
mlp_clf = MLPClassifier()
```

#### Every model is fitted to the training data.


```python
for clf in (log_clf, rnd_clf, svm_clf, kn_clf, mlp_clf, tree_clf, ovr_clf, ovr2_clf):
    clf.fit(X_train, Y_train)
    ypred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(Y_test, ypred))
```

    LogisticRegression 0.19318181818181818
    RandomForestClassifier 0.3068181818181818
    SVC 0.11363636363636363
    KNeighborsClassifier 0.20454545454545456
    MLPClassifier 0.14772727272727273
    DecisionTreeClassifier 0.11363636363636363
    OneVsRestClassifier 0.2159090909090909
    OneVsRestClassifier 0.045454545454545456
    

#### The models do not produce good accuracies exept for the Random Forest model with approximately 0.3 accuracy. As a result, to increase the model performance we will use ensemble methods and hyperparameter tuning.

## Hyperparameter Tuning

#### For Hyperparameter tuning we will use both GridSearch and RandomSearch. While grid search looks at every possible 
#### combination of hyperparameters to find the best model, random search only selects and tests a random combination of 
#### hyperparameters. ( Our first hyperparameter tuning will be performed on a Naive Bayes model).


```python
param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=40, n_jobs=-1)
nbModel_grid.fit(X_train, Y_train)
print(nbModel_grid.best_estimator_)
nb_mod_clf = nbModel_grid.best_estimator_
```

    Fitting 40 folds for each of 100 candidates, totalling 4000 fits
    GaussianNB(var_smoothing=0.12328467394420659)
    

#### Our second tuning will be performed on a KNeigbours model.


```python
metrics = ['euclidean','manhattan'] 
neighbors = np.arange(1, 16)
param_grid  = dict(metric=metrics, n_neighbors=neighbors)
```


```python
cross_validation_fold = 40
grid_search = GridSearchCV(kn_clf, param_grid, cv=cross_validation_fold, scoring='accuracy', refit=True)
grid_search.fit(X_train, Y_train)
```




    GridSearchCV(cv=40, estimator=KNeighborsClassifier(),
                 param_grid={'metric': ['euclidean', 'manhattan'],
                             'n_neighbors': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])},
                 scoring='accuracy')




```python
KN_mod_clf = grid_search.best_estimator_
```

#### Our third tuning will be performed on a Random Forest model.


```python
param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}
```


```python
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
grid_search.fit(X_train, Y_train)
print(grid_search.best_estimator_)
```

    RandomForestClassifier(max_depth=6, max_features='sqrt', max_leaf_nodes=9,
                           n_estimators=50)
    


```python
rf_mod_clf = RandomForestClassifier(max_depth=6,
                                    max_features="sqrt",
                                    max_leaf_nodes=9,
                                    n_estimators=50)
```

#### Compare accuracies between these 3 new models.


```python
for clf in (nb_mod_clf, KN_mod_clf, rf_mod_clf):
    clf.fit(X_train, Y_train)
    ypred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(Y_test, ypred))
```

    GaussianNB 0.29545454545454547
    KNeighborsClassifier 0.3068181818181818
    RandomForestClassifier 0.3522727272727273
    

#### Random Forest still outperforms the other models. As a matter of fact, hyperparameter tuning built a classification model with a 5% increase in accuracy, making it our optimal model so far.

## Enseble models


```python
voting_clf = VotingClassifier(estimators=[('kn', kn_clf),('rf', rnd_clf),('lr', log_clf)], voting='soft')
voting2_clf = VotingClassifier(estimators=[('rf',rnd_clf),('nb',nb_mod_clf),('lr', log_clf)], voting='soft')
voting3_clf = VotingClassifier(estimators=[('sv',svm_clf),('kN',KN_mod_clf),('nB',nb_mod_clf),('lr', log_clf)], voting='hard')
voting4_clf = VotingClassifier(estimators=[('rfm', rf_mod_clf),('kN',KN_mod_clf),('smv',svm_clf)], voting='hard')
```

#### Compare accuracy for the models created using the ensemble method.


```python
for clf in (voting_clf, voting2_clf, voting3_clf, voting4_clf):
    clf.fit(X_train, Y_train)
    ypred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(Y_test, ypred))
```

    VotingClassifier 0.2727272727272727
    VotingClassifier 0.29545454545454547
    VotingClassifier 0.23863636363636365
    VotingClassifier 0.3181818181818182
    

#### The best performance comes from the last ensemble model which is consisted of a Random Forest, a Naive Bayes and a KNeighbour
#### model, after performing the hyperparameter tuning.

## Optimal Model Predictions


```python
ypred = rf_mod_clf.predict(X_test)
ypred
```




    array(['album rock', 'dance pop', 'adult standards', 'adult standards',
           'dance pop', 'adult standards', 'dance pop', 'dance pop',
           'adult standards', 'album rock', 'adult standards', 'album rock',
           'adult standards', 'adult standards', 'adult standards',
           'dance pop', 'album rock', 'dance pop', 'dance pop', 'album rock',
           'dance pop', 'album rock', 'adult standards', 'album rock',
           'adult standards', 'adult standards', 'album rock', 'album rock',
           'dance pop', 'adult standards', 'adult standards', 'dance pop',
           'adult standards', 'dance pop', 'adult standards', 'dance pop',
           'adult standards', 'adult standards', 'dance pop', 'dance pop',
           'adult standards', 'adult standards', 'dance pop',
           'adult standards', 'dance pop', 'dance pop', 'dance pop',
           'album rock', 'dance pop', 'dance pop', 'adult standards',
           'dance pop', 'dance pop', 'album rock', 'dance pop', 'album rock',
           'adult standards', 'album rock', 'album rock', 'album rock',
           'adult standards', 'album rock', 'dance pop', 'adult standards',
           'dance pop', 'album rock', 'dance pop', 'adult standards',
           'dance pop', 'album rock', 'adult standards', 'dance pop',
           'album rock', 'adult standards', 'dance pop', 'adult standards',
           'dance pop', 'adult standards', 'dance pop', 'adult standards',
           'dance pop', 'dance pop', 'dance pop', 'album rock', 'dance pop',
           'album rock', 'adult standards', 'adult standards'], dtype=object)



## Testing the model on the test data


```python
test = pd.read_csv("CS98XClassificationTest.csv")
```


```python
Id_number = test.loc[:, 'Id']
```


```python
test = test.drop(['Id','title','artist'],axis=1)
```


```python
test[["acous", "spch", "live"]] = scaler.fit_transform(test[["acous", "spch", "live"]])
```

#### After importing the testing set, it is vital to perform the exact data - preprocessing to match the data format of the training set. Otherwise, our model will receive a decrease in accuracy.


```python
final_predictions = rf_mod_clf.predict(test)
```

#### Importing the predicted values to a ".csv" file. This step, of creating a .csv file, is not related with the classification project, and is only essential due to the Kaggle competition as a submission file.


```python
CSV = pd.DataFrame({
    "Id": Id_number,
    "top genre": rf_mod_clf.predict(test)
})
CSV.to_csv("classification.csv", index=False)
```

## Finally, we use our Multiclass Classification model to predict the song genres of new data and check our model's accuracy.


```python
test2 = pd.read_csv('CS98XRegressionTest.csv')
```


```python
test2 = test2.dropna()
validation_test = test2["top genre"]
```


```python
n = 0
for i in range(0,113,1):
    if list(final_predictions)[i] == list(validation_test)[i]:
        n = n + 1

print(n)
```

    36
    


```python
accuracy_score(validation_test, final_predictions)
```




    0.3185840707964602



#### A useful step is to check the Confusion matrix, in order to identify which classes were predicted accurately by our model and in which classes were the most misclassifications occured.


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(validation_test, final_predictions)
# Plotting the confusion matrix using seaborn heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```


    
![png](output_79_0.png)
    



```python
# Get the number of classes
num_classes = len(cm)

# Initialize a list to store misclassifications for each class
misclassifications_per_class = []

# Calculate misclassifications for each class
for i in range(num_classes):
    # Sum false positives (FP) and false negatives (FN) for class i
    misclassifications = cm[:, i].sum() - cm[i, i]
    misclassifications_per_class.append(misclassifications)

# Print misclassifications for each class with misclassifications
for i, misclassifications in enumerate(misclassifications_per_class):
    if misclassifications > 0:
        print(f"Class {i}: {misclassifications} misclassifications")
```

    Class 0: 20 misclassifications
    Class 1: 27 misclassifications
    Class 20: 30 misclassifications
    


```python
validation_test.unique()
```




    array(['dance pop', 'glam rock', 'big beat', 'appalachian folk',
           'adult standards', 'mellow gold', 'album rock',
           'brill building pop', 'barbadian pop', 'british invasion',
           'bubblegum dance', 'hollywood', 'cowboy western', 'hip hop',
           'g funk', 'eurodance', 'native american', 'alternative country',
           'east coast hip hop', 'art rock', 'blues rock', 'dance rock',
           'classic country pop', 'beach music', 'neo mellow', 'disco',
           'europop', 'classic rock', 'bronx hip hop', 'alternative metal',
           'big room', 'modern rock', 'dirty south rap', 'canadian pop',
           'boy band', 'deep adult standards', 'diva house', 'jazz fusion',
           'glam metal'], dtype=object)



#### We identify that the classes of "dance pop", "glam rock" and "art rock" are often misclassified by our model.

# Predict the housing price

Use the following steps to build/test the model

## 1. Understand what problem we are trying to solve
- Objective: Be able to forcast the housing price in CA using Regression algorithm
 
- How to measure the performance
	RMSE (Root Mean Squaure Error)
	 MAE (Mean Absolute Error) 
	
	Typically, RMSE used but when there are many outliers in the dataset, then MSE is preferred because RMSE is more sensitive to outliers than the MAE
    
![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Froad2roam%2F1N9iLtZFGd.png?alt=media&token=3f43264e-7de6-4c50-be95-9b9797669787)

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Froad2roam%2FWXJVo4qeKv.png?alt=media&token=fb1cee01-3326-4ce1-bfef-076ade14304b)

- Assumption : list out assumption used to build the model from the dataset

## 2. Collect Data
- numerical field
- categorical field
- split training and test data

## 3. Explore Data
- Correlation analysis
- Attribute combination

## 4. Prepare Data
#### 4.1 Data cleaning for numerical data
- Missing values: replace with median value
	
	```python
	from sklearn.impute import SimpleImputer
	imputer = SimpleImputer(strategy='median')
	```
	
- Before applying the median imputer, create a dataset with numerical values only first then fit 
- `housing_num` is DataFrame
	
	```python
	housing_num = housing.drop(columns=['ocean_proximity'])
	```
	
- Compute the median values for each of the numerical data columns. See print the computed meidan, use `statistics_`

	```python
	imputer.fit(housing_num)
	imputer.statistics_
	```

- Apply the median values to the missing values using `transform` function
- Then convert the np array type to DF
- `X` is np array type
	```python
	X = imputer.transform(housing_num)
	housing_train = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
	```
	
	
#### 4.2 Data cleaning for text and categorical data
- There is only one attribute in this dataset, 'ocean_proximity'
	```python
	housing_cat = housing['ocean_proximity']  #: this will return Panda series
	housing_cat = housing[['ocean_proximity']] #: this will return Pandas DF
	
	```
- ML algoirthms prefer to work with numerical data, let's convert the categorical data from text to numbers using Scikit-Learn's OrdinalEncoder class. 
- Use `fit_transform` method to combin `fit` and `transform` methods into one
	```python
	
	from sklearn.preprocessing import OrdinalEncoder
	ordinal_encoder = OrdinalEncoder()
	housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
	```
	
- Use Onehot encoding to convert categorical data to number. Unlike ordinal encorder no value assiciating with number but more features will be generated.
	```python
	from sklearn.preprocessing import OneHotEncoder
	onehot_encoder = OneHotEncoder()
	housing_cat_1hot = onehot_encoder.fit_transform(housing_cat)
	```
	
- OneHotEncoder will return sparse matrix to convert to array
	```python
	housing_cat_1hot.toarray()
	```
	
- One-hot encoding will result in a large number of input features. This may slow down. Then you may want to replace the categorical input with numerical features related to the categoreis
	- for example Ocean_proximity can be replaced with the distance to the ocean.
	
#### 4.3 Custom features
- By creating a custom transformer, we can generate combination features 
	```python
	from sklearn.base import BaseEstimator, TransformerMixin

	rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

	class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
		def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
			self.add_bedrooms_per_room = add_bedrooms_per_room

		def fit(self, X, y=None):
			return self #: nothing to do

		def transform(self, X):
			rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
			population_per_household = X[:, population_ix] / X[:, households_ix]
			if self.add_bedrooms_per_room:
				bedrooms_per_room = X[: bedrooms_ix] / X[:, rooms_ix]
				return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
			else:
				return np.c_[X, rooms_per_household, population_per_household]

		def fit_transform(self, X):
			return self

	attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
	housing_extra_attribs = attr_adder.transform(housing.values)
	```

#### 4.4 Feature Scaling
- min-max scaling (normalization) values between 0 and 1
	- Scikit-Learn's MinMaxScaler
- standardization
	- Less affected by outliers
	- Scikit-Learn's StandardScaler

#### 4.5 Build Transformation Pipelines
- Using Scikit-Learn Pipeline class to put the transformation togeter
- The pipeline for the numerical features
	```python
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	
	num_pipeline = Pipeline ([
		('imputer', SimpleImputer(strange='median')),
		('attribs_adder', CombinedAttributesAdder()),
		('std_scaler', StandardScaler()),
		])
	housing_num_tr = num_pipeline.fit_transform(housing_num)
	```

- Combine the numerical feature pipeline and the categorical feature pipeline using ColumnTransformer
	```python
	from sklearn.compose import ColumnTransformer
	num_attribs = list(housing_num)
	cat_attribs = ['ocean_proximity']

	full_pipeline = ColumnTransformer([
    	('num', num_pipeline, num_attribs),
    	('cat', OneHotEncoder(), cat_attribs)
	])

	housing_prepared = full_pipeline.fit_transform(housing)
	```


## 5. Train Model

#### 5.1 Training and Evaluating on the Training Set

- Train a Linear Regression model
	```python
	from sklearn.linear_model import LinearRegression

	lin_reg = LinearRegression()
	lin_reg.fit(housing_prepared, housing_labels)
	```

- Measure the Linear Regression model using RMSE
	```python
	from sklearn.metrics import mean_squared_error
	housing_prediction = lin_reg.predict(housing_prepared)
	lin_mse = mean_squared_error(housing_labels, housing_prediction)
	lin_rmse = np.sqrt(lin_mse)
	```

- median_housing_values range between 120K and 265K. Getting RMSE error of 68K not very good model.
- To imporve model, we could do 
	- choose more powerful model
	- improve the feature data
	- reduce the constrains on the model (if regularization is used, don't use it)

- Try more complex model that allows non-linear relationship such as DecisionTreeRegressor
	```python
	from sklearn.tree import DecisionTreeRegressor
	tree_reg = DecisionTreeRegressor()
	tree_reg.fit(housing_prepared, housing_labels)
	```
	
- Mesure the Decision Tree Regression model using RMSE
	```python
	housing_prediction = tree_reg.predict(housing_prepared)
	tree_mse = mean_squared_error(housing_labels, housing_prediction)
	tree_rmse = np.sqrt(tree_mse)
	tree_rmse
	```

- The RMSE is 0.0. The model has badly overfit the data!

#### 5.2 Apply Cross-Validation to prevent overfitting
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
				scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
print('Scores:', tree_rmse_scores)
print('Mean:', tree_rmse_scores.mean())
print('STD:',tree_rmse_scores.std())
```
	
Scores: [68247.16483949 66971.30655813 70595.82761205 68775.26697836
				70616.41405011 75011.35906579 69979.94022458 71242.9361998
				78097.76485763 69144.56207634]
Mean: 70868.25424622856
STD: 3160.795638931402

## 6. Optimize Model

#### 6.1 Grid Search to find the optimal hyperparameters
```python
from sklearn.model_selection import GridSearchCV
param_grid = [
	{'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
	{'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
						  scoring='neg_mean_squared_error',
						  return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

#### 6.2 Evaluate the Test dataset
```python
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```
## 7. Depoly Model


REFERENCE: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron




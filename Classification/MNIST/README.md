# Binary Classifier using MNIST dataset

## 1. Problem
- Build a binary classifier to classify MNIST data as number 5 or NOT

## 2. Collect Data
- Sklearn prvides the dataset

## 3. Exploer Data


## 4. Prepare Data


## 5. Train Model
- Performance measure

- Cross Validation
	- in the Regression problem, we used cross_val_score() method in SciKit Learn, for the Classfification problem, we will use StratifiedKFold as default. See the code below
	```python
	from sklearn import datasets
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


	data = datasets.load_breast_cancer()
	x, y = data.data, data.target

	print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=5))
	print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=KFold(n_splits=5)))
	print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=StratifiedKFold(n_splits=5)))
	```

[0.90434783 0.90434783 0.92035398 0.94690265 0.91150442]
[0.89473684 0.92982456 0.94736842 0.95614035 0.82300885]
[0.90434783 0.90434783 0.92035398 0.94690265 0.91150442]


## 6. Optimize Model



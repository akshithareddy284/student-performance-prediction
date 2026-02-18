import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_PATH = BASE_DIR / "student-por.csv"

data = pd.read_csv(r"student-por.csv", sep=";")
print(data.head())
print(data.columns)
print(data.shape)
print("Avg of G3:",data["G3"].mean())
print("10percent of Avg of G3:",(data["G3"].mean()*0.1))
print("15percent of Avg of G3:",(data["G3"].mean()*0.15))


##data preprocessing steps
##missing values
##no missing values

#EDA
import matplotlib.pyplot as plt
import seaborn as sns

#Distribution of Target Variable G3
sns.histplot(data["G3"], bins=20, kde=True)
plt.title("Distribution of Final Grades (G3)")

# Relationships with G3 

sns.scatterplot(x=data["G1"], y=data["G3"], label="G1 vs G3")
plt.show()
sns.scatterplot(x=data["G2"], y=data["G3"], label="G2 vs G3")
plt.show()

plt.title("G1 & G2 vs G3")
plt.legend()
plt.show()

##check duplicates
print(data.duplicated().sum())

##separate the target variable
Y=data["G3"]
print(Y)

X = data[["G2", "G1", "absences", "failures", "studytime"]]
print(X)

##correlation
corr_data = X.copy()
corr_data["G3"] = Y
correlation = corr_data.corr()["G3"].sort_values(ascending=False)
print(correlation)

##train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

##hyperparamter tuning
##Random forest regressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 3, 5],
}


grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)

forest = grid.best_estimator_

y_pred = forest.predict(x_test)
print(y_pred)

##comparision
comparison = pd.DataFrame({
    "Actual G3": y_test.values,
    "Predicted G3": y_pred
})
print(comparison.head(10))

##errror for the target variable G3
comparison["Error"] = comparison["Actual G3"] - comparison["Predicted G3"]
print(comparison.head(10))

##evaluation metrics
from sklearn.metrics import r2_score,root_mean_squared_error
r2=r2_score(y_test,y_pred)
print(" R2 score:",r2)

rmse=root_mean_squared_error(y_test,y_pred)
print("rmse score:",rmse)

# Adjusted R2
n = x_test.shape[0]   # number of samples
p = x_test.shape[1]   # number of features

adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print("Adjusted R2:", adjusted_r2)

##check overfitting
train_r2 = forest.score(x_train, y_train)
test_r2 = forest.score(x_test, y_test)

print("Train R2:", train_r2)
print("Test R2:", test_r2)

##feature importance
importance = pd.Series(forest.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print(importance.head(10))

##bar plot of feature importance

plt.figure(figsize=(8,5))
importance.plot(kind='bar')

plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import pickle
pickle.dump(forest, open("student_model.pkl", "wb"))


























import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

data=fetch_california_housing()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)

x = x[["MedInc", "AveRooms", "AveOccup", "HouseAge"]]  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred_lin=lin_reg.predict(x_test)

ridge_reg=Ridge(alpha=1.0)
ridge_reg.fit(x_train,y_train)
y_pred_ridge=ridge_reg.predict(x_test)

poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)

poly_reg=LinearRegression()
poly_reg.fit(x_train_poly,y_train)
y_pred_poly=poly_reg.predict(x_test_poly)

def evaluate_model(name,y_true,y_pred):
    print(f"{name} Regression")
    print("MSE: ",mean_squared_error(y_true,y_pred))
    print("R2 Score:",r2_score(y_true,y_pred))
    print("-" * 30)

evaluate_model("Linear",y_test,y_pred_lin)
evaluate_model("Ridge:",y_test,y_pred_ridge)
evaluate_model("Polynomial:",y_test,y_pred_poly)

sort_idx = np.argsort(x_test["MedInc"].values.flatten())
plt.plot(x_test["MedInc"].values.flatten()[sort_idx], y_pred_lin[sort_idx], color='blue', label='Linear')
plt.plot(x_test["MedInc"].values.flatten()[sort_idx], y_pred_ridge[sort_idx], color='red', label='Ridge')
plt.plot(x_test["MedInc"].values.flatten()[sort_idx], y_pred_poly[sort_idx], color='green', label='Polynomial')

plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("House Price Prediction Comparison (Using Multiple Features)")
plt.legend()
plt.show()





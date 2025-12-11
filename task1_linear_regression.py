import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("train.csv")


features = ['GrLivArea', 'FullBath', 'BedroomAbvGr']

X = df[features]
y = df['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("RMSE:", rmse)


plt.figure(figsize=(15, 7))
plt.scatter(y_test, y_pred, alpha=0.6, s=100)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=3
)
plt.xlabel("Actual Price", fontsize=15)
plt.ylabel("Predicted Price", fontsize=15)
plt.title("Actual vs Predicted House Prices", fontsize=18)
plt.grid(True)
plt.show()


plt.figure(figsize=(15, 7))

# scatter
plt.scatter(df['GrLivArea'], df['SalePrice'],
            alpha=0.4, s=60, label="Actual Data")

# line
x_line = np.linspace(df['GrLivArea'].min(), df['GrLivArea'].max(), 200)
y_line = model.coef_[0] * x_line + model.intercept_

plt.plot(x_line, y_line, linewidth=3, label="Regression Line")

plt.xlabel("Living Area (sq ft)", fontsize=15)
plt.ylabel("Sale Price", fontsize=15)
plt.title("GrLivArea vs SalePrice (Trend Line)", fontsize=18)
plt.legend()
plt.grid(True)
plt.show()


test_df = pd.read_csv("test.csv")
test_df = test_df.fillna(0)

test_pred = model.predict(test_df[features])

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created!")

plt.figure(figsize=(10,5))
plt.scatter(df["GrLivArea"], df["SalePrice"])
plt.title("Living Area (sq ft) vs Sale Price")
plt.xlabel("GrLivArea (sq ft)")
plt.ylabel("Sale Price")
plt.show()


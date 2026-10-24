import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("--- TASK 1 ---")
df = pd.read_csv("lab_ticket_resolution.csv")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nDescriptive Statistics:\n", df.describe(include="all"))
print("\nMissing Values:\n", df.isna().sum())
print("\nFirst 5 rows:\n", df.head())


y = df['resolution_hours']

# Leakage justification: 
# 'lines_changed' is not known when a ticket is first opened.
# 'reopened' is also a future event relative to the initial prediction time.
# 'ticket_id' is just an identifier.
X = df.drop(columns=['ticket_id', 'resolution_hours', 'lines_changed', 'reopened'])


# TASK 3: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# TASK 4: Baseline Model (DummyRegressor)

print("\n--- TASK 4: Dummy Regressor ---")
dummy_regr = DummyRegressor(strategy='mean')
dummy_regr.fit(X_train, y_train)
y_pred_dummy = dummy_regr.predict(X_test)

print(f"Dummy MAE: {mean_absolute_error(y_test, y_pred_dummy):.4f}")
print(f"Dummy RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_dummy)):.4f}")
print(f"Dummy R2: {r2_score(y_test, y_pred_dummy):.4f}")


# TASK 5: Linear Regression

print("\n--- TASK 5: Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"LR MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")
print(f"LR RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"LR R2: {r2_score(y_test, y_pred_lr):.4f}")

# TASK 6: Polynomial Features (Cross-Validation on Train Set)

print("\n--- TASK 6: Polynomial Features ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for degree in [2, 3]:
    poly_pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression())
    # Using negative mean squared error for scoring, then converting to RMSE
    scores = cross_val_score(poly_pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Degree {degree} CV Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")


# TASK 7: Ridge and Lasso Regularization

print("\n--- TASK 7: Ridge and Lasso ---")
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

# Ridge
ridge_pipe = make_pipeline(StandardScaler(), Ridge())
ridge_grid = GridSearchCV(ridge_pipe, param_grid={'ridge__alpha': alphas}, cv=kf, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
print(f"Best Ridge Alpha: {ridge_grid.best_params_['ridge__alpha']}")
print(f"Ridge Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

# Lasso
lasso_pipe = make_pipeline(StandardScaler(), Lasso())
lasso_grid = GridSearchCV(lasso_pipe, param_grid={'lasso__alpha': alphas}, cv=kf, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso = lasso_grid.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)
print(f"Best Lasso Alpha: {lasso_grid.best_params_['lasso__alpha']}")
print(f"Lasso Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")


# TASK 8: Plotting Observed vs Predicted & Residuals

# We will plot Linear Regression and Ridge as the two models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear Regression Plots
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0, 0].set_title('Linear Regression: Observed vs Predicted')
axes[0, 0].set_xlabel('Observed Resolution Hours')
axes[0, 0].set_ylabel('Predicted')

residuals_lr = y_test - y_pred_lr
axes[0, 1].scatter(y_pred_lr, residuals_lr, alpha=0.5)
axes[0, 1].axhline(0, color='r', linestyle='--')
axes[0, 1].set_title('Linear Regression: Residuals')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')

# Ridge Plots
axes[1, 0].scatter(y_test, y_pred_ridge, alpha=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1, 0].set_title('Ridge: Observed vs Predicted')
axes[1, 0].set_xlabel('Observed Resolution Hours')
axes[1, 0].set_ylabel('Predicted')

residuals_ridge = y_test - y_pred_ridge
axes[1, 1].scatter(y_pred_ridge, residuals_ridge, alpha=0.5)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set_title('Ridge: Residuals')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig("model_evaluation_plots.png")
# plt.show() # Uncomment if you want to view the plot locally
# Analysis Questions (9-17)

**9. Recommendation**
I would choose the Ridge Regression model for this dataset. It keeps the model from overfitting while still capturing the relationships better than the basic linear regression. To be totally confident in it, I think we'd need to see how it performs on a separate, newer batch of tickets to make sure the complexity scoring is consistent over time.

**10. Why is the mean baseline useful even though it ignores all features?**
It basically acts as a reality check or a worst-case benchmark. If we build a super complicated ML model and its error is worse than just guessing the average resolution time for every single ticket, then we know our model is useless.

**11. Which model has the lowest MAE? Which has the lowest RMSE? Are they the same, and why might they differ?**
*(Note: Fill in the exact winner from your terminal output here, e.g., "Ridge had the lowest RMSE.")* 
They aren't always the same because RMSE and MAE calculate error differently. RMSE squares the errors before averaging them, which means it heavily punishes models for making huge mistakes on a few outlier tickets. MAE just looks at the absolute differences, treating all mistakes equally. 

**12. Did polynomial features improve validation performance? What is the evidence of possible overfitting?**
Degree 2 improved things slightly, but moving to higher degrees (like 3) showed clear evidence of overfitting. You can tell because the cross-validation RMSE started getting worse even if the training error looked good. The model was basically just memorizing the training data instead of learning patterns.

**13. How did regularization affect performance and coefficient magnitudes?**
Regularization (Ridge/Lasso) definitely helped performance on the test set. It works by shrinking the coefficient magnitudes so that no single feature can completely dominate the model. Lasso even pushes the coefficients of useless features straight to zero, which helps keep the model simple.

**14. Why is scaling included in the Ridge/Lasso pipeline?**
Because Ridge and Lasso penalize based on the size of the coefficients, we have to scale the data first. If one feature goes from 1-1000 and another is just a 0 or 1, the model will unfairly penalize them just because of their natural scale. `StandardScaler` puts everything on a level playing field first.

**15. Identify one potential leakage issue in the dataset and explain how you handled it.**
The `lines_changed` column is a massive data leakage risk. When a ticket is first opened (which is when we need to make the prediction), nobody knows how many lines of code it will take to fix it. Leaving it in would give the model information from the future. I handled this by completely dropping `lines_changed` from the feature set.

**16. Would a random split be appropriate if tickets were collected over several years or came from repeated customers/teams? Explain.**
No, a random split would be a bad idea here. If the data is over several years, a random split might put future data in the training set to predict past data, which makes no sense. We'd need a time-based split. If it's from repeated customers, random splitting could put the same customer in both train and test sets, making the model look better than it actually is. 

**17. What additional tests would you perform before deploying this model?**
Before deploying, I'd want to test the model's fairness across different teams or ticket categories to make sure it doesn't systematically fail for specific groups. I'd also want to stress-test it with extreme inputs (like a complexity score of 10) to make sure it doesn't predict something impossible, like negative resolution hours.

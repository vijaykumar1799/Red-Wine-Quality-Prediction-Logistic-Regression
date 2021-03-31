# Red-Wine-Quality-Prediction-Logistic-Regression

- Problem: Having the red wine quality dataset, the goal is to predict whether a specific wine with the defined features is good or bad.
  Since the dataset contains grades ranging from 0 upto 10, for a wine to be classified as good it should score 7 or greater and anything
  less than 7 is a bad quality wine.
  
- Solution: To Solve this problem, the dataset is modified in sucha way that every grade that is greater or equal to 7 holds the value 1 
  which means good quality and 0 for bad quality.
  
- Algorithm used: Since this is the case of a binary classification, Logistic regression is used and the weights are optmized using the 
  Batch Gradient Descent Algorithm.
  
- Results: As results, the logistic regression model, predicts correctly with an accuracy of 86.87% and RMSE of 0.36.

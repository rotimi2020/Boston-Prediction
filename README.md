## ****Title: Boston Housing Data****
1. Title: Boston Housing Data

2. Sources:
   (a) Origin:  This dataset was taken from the StatLib library which is
                maintained at Carnegie Mellon University.
   (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the 
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102, 1978.
   (c) Date: July 7, 1993

    Download file : https://github.com/rotimi2020/Boston-Prediction/blob/main/Boston.csv
            
3. Relevant Information:

   Concerns housing values in suburbs of Boston.

4. Number of Instances: 506

5. Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.

6. Attribute Information:
* CRIM: capita crime rate by town
* ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS: proportion of non-retail business acres per town 
* CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise) 
* NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
* RM: average number of rooms per dwelling 
* AGE: proportion of owner-occupied units built prior to 1940
* DIS: weighted distances to five Boston employment centres
*  RAD: index of accessibility to radial highways 
*  TAX: full-value property-tax rate per $10,000 [$/10k] 
*  PTRATIO: pupil-teacher ratio by town 
*  B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
*  LSTAT: % lower status of the population 
* MEDV: Median value of owner-occupied homes in $1000's [k$]
    
7. Missing Attribute Values:  None.



 ***Boxplot Diagram*** ****EDA***

* ****Linear/Non Linear Algorithms***. 
1. Linear Regression
2. Lasso
3. ElasticNet
4. Decision Tree Regression
5. KNeighbors Regression
6. SVR

* ****Ensemble Algorithms***. 
1. AdaBoost Regression
2. ExtraTrees Regression
3. Random Forest Regression
4. Gradient Boosting Regression
5. Light GradientBoosting Regression
6. CatBoost Regression 

* The Boxplot show Lasso and ElasticNet with very low mean score (neg_mean_absolute_error)
* All ensemble ALgorithm show a promising and high mean score (neg_mean_absolute_error)
* Hence Ensemble algorithm should be use for my predictive modelling
* 

#### The final best model is CatBoost Regression

---------------------------------
#### Best Model : -------- CatBoost Regressor -------- 
---------------------------------
* Best: -0.098864 using {'learning_rate': 0.01}
---------------------------------
Test set evaluation:

--------------
* MAE: 0.08106293221319592
--------------
* MSE: 0.009480761870486888
--------------
* RMSE: 0.09736920391215535
--------------
* R2 Square 0.9220099396058459
--------------
Train set evaluation:

--------------
* MAE: 0.05315783964480034
--------------
* MSE: 0.004942925912769483
--------------
* RMSE: 0.07030594507415061
--------------
* R2 Square 0.9592019663617606
--------------

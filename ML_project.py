import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_excel(r'C:\Users\jesse\Documents\Machine Learning D\ML Project\Data\final_data.xlsx') #read and stores the data
X,y = data['Air temperature'].to_numpy().reshape(-1,1), data['Deaths'].to_numpy()
X_kfold, X_test, y_kfold, y_test = train_test_split(X,y,test_size=0.25, random_state=50) #Split the dataset into K-Fold and testing datasets

#visualizing the data
plt.figure(figsize=(8,6)) #create a figure
plt.scatter(X, y, color='b', s=12) # each datapoint is depicted by a dot in color 'blue' and size '12'
plt.xlabel('Average temperature',size=15) # define label for the horizontal axis
plt.ylabel('Deaths',size=15) # define label for the vertical axis
plt.title('Relation between deaths and average temperature',size=15) # define the title of the plot
plt.show()

#visualizing the data
plt.figure(figsize=(8,6)) #create a figure
plt.scatter(X, data['Maximum temperature'], color='b', s=12) # each datapoint is depicted by a dot in color 'blue' and size '12'
plt.scatter(X, data['Minimum temperature'], color='r', s=12) # each datapoint is depicted by a dot in color 'red' and size '12'
plt.xlabel('Average temperature',size=15) # define label for the horizontal axis
plt.ylabel('Max and Min temperature',size=15) # define label for the vertical axis
plt.title('Relation between Max, Min and average temperature',size=15) # define the title of the plot
plt.show()

#visualizing the data
plt.figure(figsize=(8,6)) #create a figure
plt.scatter(data['Precipitation amount'], y, color='b', s=12) # each datapoint is depicted by a dot in color 'blue' and size '12'
plt.xlabel('Precipitation amount',size=15) # define label for the horizontal axis
plt.ylabel('Deaths',size=15) # define label for the vertical axis
plt.title('Relation between Precipitation amount and Deaths',size=15) # define the title of the plot
plt.show()

data = data.drop(['TimeWeeks','Precipitation amount','Maximum temperature','Minimum temperature'], axis=1) #removes the colums 'TimeWeeks','Precipitation amount','Maximum temperature','Minimum temperature' that are not used

# Calculates training and validation errors using K-Fold method. This knowledge is used to decide which polynomic degree we want to choose for our final model.
n_spilts_plo = 5 #number of splits
kfold1 = KFold(n_splits=n_spilts_plo, shuffle=True, random_state=40)
degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10] #degree cancidates
tr_errors_pol = {}
val_errors_pol = {}

for i, degree in enumerate(degrees):
    tr_errors_pol[degree] = []
    val_errors_pol[degree] = []
    for train_index, test_index in kfold1.split(X_kfold):
        # Define the training and validation data using the indices returned by kfold and numpy indexing
        X_train, X_val = X_kfold[train_index], X_kfold[test_index]
        y_train, y_val = y_kfold[train_index], y_kfold[test_index]

        # preparing polynomial regression
        lin_reg = LinearRegression(fit_intercept=False)
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        lin_reg.fit(X_train_poly, y_train)

        #calculating training and validation errors
        y_pred_train = lin_reg.predict(X_train_poly)
        tr_error = mean_squared_error(y_train, y_pred_train)
        X_val_poly = poly.transform(X_val)
        y_pred_val = lin_reg.predict(X_val_poly)
        val_error = mean_squared_error(y_val, y_pred_val)

        #storing the values
        tr_errors_pol[degree].append(tr_error)
        val_errors_pol[degree].append(val_error)

#calculates average training and validation errors for each degree
average_tr_errors = []
average_val_errors = []
for key in degrees:
    tr_list = tr_errors_pol[key]
    val_list = val_errors_pol[key]
    average_tr_error = sum(tr_list)/len(tr_list)
    average_val_error = sum(val_list)/len(val_list)
    average_tr_errors.append(average_tr_error)
    average_val_errors.append(average_val_error)

#Plots average training and validation errors for each degrees
plt.figure(figsize=(8, 6))
plt.plot(degrees, average_tr_errors, label = 'Average training error')
plt.plot(degrees, average_val_errors,label = 'Average validation error')
plt.legend(loc = 'upper left')
plt.xlabel('Degree')
plt.ylabel('Loss')
plt.title('Train vs validation loss for polynomial regression')
plt.show()

degree = 7 #(best) degree for the polynomial regression
tr_errors_pol_ = {}
val_errors_pol_ = {}
tr_errors_lin_ = {}
val_errors_lin_ = {}
opt_intercept_lin_ = {}
opt_coefficient_lin_ = {}
i = 0
plt.figure(figsize=(9,9)) #create a figure
n_splits = 5 #number of K-fold splits
kfold2 = KFold(n_splits=n_splits, shuffle=True, random_state=40) #Defining the kfold object that will use for cross validation

for train_index, test_index in kfold2.split(X_kfold):
    # Define the training and validation data using the indices returned by kfold and numpy indexing
    X_train, X_val = X_kfold[train_index], X_kfold[test_index]
    y_train, y_val = y_kfold[train_index], y_kfold[test_index]

    #preparing linear regression
    reg_lin = LinearRegression(fit_intercept=True)
    reg_lin.fit(X_train,y_train)

    y_pred_train_lin = reg_lin.predict(X_train)  # predict training label values
    tr_error_lin = mean_squared_error(y_train, y_pred_train_lin)  # training error
    y_pred_val_lin = reg_lin.predict(X_val)  # predict validation label values
    val_error_lin = mean_squared_error(y_val, y_pred_val_lin)  # validation error

    # preparing polynomial regression
    reg_lin_pol = LinearRegression(fit_intercept=False)
    reg_pol = PolynomialFeatures(degree=degree) #generate polynomial feature
    X_train_pol = reg_pol.fit_transform(X_train) #fit the raw features
    reg_lin_pol.fit(X_train_pol, y_train) #apply linear regression

    y_pred_train_pol = reg_lin_pol.predict(X_train_pol)  #predict using linear model
    tr_errors_pol = mean_squared_error(y_train, y_pred_train_pol)  #training error

    X_val_pol = reg_pol.fit_transform(X_val) #fit the raw features
    y_pred_val_pol = reg_lin_pol.predict(X_val_pol) #predict using linear model
    val_errors_pol = mean_squared_error(y_val, y_pred_val_pol) #validation error

    # storing the values
    tr_errors_lin_[i] = tr_error_lin
    val_errors_lin_[i] = val_error_lin
    opt_intercept_lin_[i] = reg_lin.intercept_
    opt_coefficient_lin_[i] = reg_lin.coef_

    tr_errors_pol_[i] = tr_errors_pol
    val_errors_pol_[i] = val_errors_pol

    if i in [0, 2, 4]:
        plt.subplot(3, 2, i + 1)  # choose the subplot
        plt.tight_layout()
        x_fit = np.linspace(-25, 25, 100) #generate samples
        plt.plot(x_fit, reg_lin.predict(x_fit.reshape(-1, 1)), label='Linear model') #plots the linear regression
        plt.plot(x_fit, reg_lin_pol.predict(reg_pol.transform(x_fit.reshape(-1, 1))), label="Polynomic model") #plots the polynomical regression
        plt.scatter(X_train, y_train, color='b', s=10, label='Train datapoints') #plots training data
        plt.scatter(X_val, y_val, color='r', s=10, label='Validation datapoints') #plots validatino data
        plt.xlabel('Average temperature') #plots x-label
        plt.ylabel('Deaths') #plots y-label
        plt.legend(loc='best') #plots legend
        plt.title(f'Linear and Polynomic Regression\nCV iteration = {i+1}') #plots title
        #plt.text(25, 0, f'Linear training error = {tr_error_lin}\nPolynomic training error = {tr_errors_pol}\nLinear validation error = {val_error_lin}\nPolynomic validation error = {val_errors_pol}')
    i += 1
plt.show() #show the plot

#calculating testing errors for both models
y_pred_test_lin = reg_lin.predict(X_test)
test_error_lin = mean_squared_error(y_test, y_pred_test_lin)
X_test_pol = reg_pol.fit_transform(X_test)
y_pred_test_pol = reg_lin_pol.predict(X_test_pol)
test_error_pol = mean_squared_error(y_test, y_pred_test_pol)

print(test_error_lin, test_error_pol)

#plots the final models
plt.figure(figsize=(10,10))
x_fit = np.linspace(-25, 25, 100)
plt.plot(x_fit, reg_lin.predict(x_fit.reshape(-1, 1)), label='Linear model')  # plots the linear regression
plt.plot(x_fit, reg_lin_pol.predict(reg_pol.transform(x_fit.reshape(-1, 1))), label="Polynomial model (degree = 7)")
plt.scatter(X_test, y_test, color='b', s=10, label='Test datapoints')  # plots training data
plt.scatter(X_train,y_train, color='r', s=10, label='Train datapoints')
plt.xlabel('Average temperature')  # plots x-label
plt.ylabel('Deaths')
plt.legend(loc='best')
plt.title(f'Linear and Polynomial model\nLinear testing error = {test_error_lin}\nPolynomial testing error = {test_error_pol}')
plt.show()

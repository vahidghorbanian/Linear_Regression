# import
import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

# Define functions
def single_LinReg():
    # Generate random data-set
    nSample = 100
    x = np.random.rand(nSample, 1)
    y = 2 + 3 * x + np.random.rand(nSample, 1)

    # sckit-learn implementation

    # Model initialization
    regression_model = LinearRegression()
    # Fit data (train model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(x)

    # Model evaluation
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)

    # Print values
    print('Slope:', regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # Plotting values

    # Data points
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')

    # Predicted values
    plt.plot(x, y_predicted, color='r')
    plt.show()

def multi_LinReg():
    # Generate data-set
    Stock_Market = {
        'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016,
                 2016, 2016, 2016, 2016, 2016, 2016, 2016],
        'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75,
                          1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
        'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1,
                              6.1, 6.1, 5.9, 6.2, 6.2, 6.1],
        'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943,
                              958, 971, 949, 884, 866, 876, 822, 704, 719]
    }
    df = DataFrame(Stock_Market, columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price'])
    print(df)


    X = df[['Interest_Rate', 'Unemployment_Rate']]
    Y = df['Stock_Index_Price']

    # with sklearn
    regression_model = LinearRegression(fit_intercept=True, normalize =False)
    regression_model.fit(X, Y)

    # Predict
    y_predicted = regression_model.predict(X)

    # Model evaluation
    rmse = mean_squared_error(Y, y_predicted)
    r2 = r2_score(Y, y_predicted)

    # Print values
    # print(y_predicted)
    print('Slope:', regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # Plot
    # Plot x1-y
    f1 = plt.figure(1)
    plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')
    plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
    plt.xlabel('Interest Rate', fontsize=14)
    plt.ylabel('Stock Index Price', fontsize=14)
    plt.grid(True)
    # Plot x2-y
    f2 = plt.figure(2)
    plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')
    plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
    plt.xlabel('Unemployment Rate', fontsize=14)
    plt.ylabel('Stock Index Price', fontsize=14)
    plt.grid(True)
    # Scatter Plot x1-x2-y
    f3 = plt.figure(3)
    ax = f3.add_subplot(111, projection='3d')
    ax.scatter(df['Interest_Rate'], df['Unemployment_Rate'], df['Stock_Index_Price'], c='r', marker='o')
    # Scatter Plot x1-x2-y_predicted
    ax.scatter(df['Interest_Rate'], df['Unemployment_Rate'], y_predicted, c='b', marker='*')
    plt.show()

def f(x):

    return x * np.sin(x)

def single_PolyReg():
    # generate points used to plot
    x_plot = np.linspace(0, 10, 100)

    # generate points and keep a subset of them
    x = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    rng.shuffle(x)
    x = np.sort(x[:20])
    y = f(x)

    # create matrix versions of these arrays
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
             label="ground truth")
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

    for count, degree in enumerate([3, 4, 5]):
        '''
        Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state = None)[source]
        Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
        '''
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1))
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
                 label="degree %d" % degree)

    plt.legend(loc='lower left')
    plt.show()

def genSample(nSample):
    np.random.seed(0)
    x = np.linspace(-20, 5, nSample) + 10 * np.random.normal(0, 1, nSample)
    y = np.linspace(-5, 15, nSample) - 1 * np.random.normal(0, 1, nSample)
    # Z = 2 * np.linspace(1, 20, nSample) * np.power(x, 3) + 5 * np.linspace(1, 20, nSample) * np.power(y, 1)
    Z = 2 * np.linspace(1, 20, nSample) * np.power(x, 3) + 2 * np.linspace(1, 20, nSample) * np.exp(y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x, y, Z


def multi_PolyReg():
    # Initialize
    sample = np.linspace(50, 2000, 100, True, False, int)
    nSample_test = 400
    tolerance = 1e-3
    pDegree = [7]
    alpha = [0]
    result = {'numSample': [], 'Poly Degree': [], 'Alpha': [],
              'intercept': [], 'RMSE': [], 'R2': [], 'R2_test': []}

    count = 1
    for nSample in sample:
        # Generate training samples
        x, y, z = genSample(nSample)

        # Generate plot samples
        x_test, y_test, z_test = genSample(nSample_test)

        for count1, polyDegree in enumerate(pDegree):
            # Create regression input
            X = PolynomialFeatures(degree=polyDegree).fit_transform(np.concatenate((x, y), axis=1))

            # Create plotting input
            X_test = PolynomialFeatures(degree=polyDegree).fit_transform(np.concatenate((x_test, y_test), axis=1))

            # Fit model
            for count2, Alpha in enumerate(alpha):
                model = Ridge(alpha=Alpha, fit_intercept=True, normalize=False, copy_X=True,
                              max_iter=None, tol=tolerance)
                model.fit(X, z)
                intercept = model.intercept_
                slope = model.coef_
                z_prediction = model.predict(X)
                z_prediction_test = model.predict(X_test)

                # Model evaluation
                rmse = np.sqrt(mean_squared_error(z, z_prediction))
                r2 = r2_score(z, z_prediction)
                r2_test = r2_score(z_test, z_prediction_test)

                # save results
                result['numSample'].append(nSample)
                result['Poly Degree'].append(polyDegree)
                result['Alpha'].append(Alpha)
                result['intercept'].append(intercept)
                result['RMSE'].append(rmse)
                result['R2'].append(r2)
                result['R2_test'].append(r2_test)

                print('Model',count, 'done!\n')
                count = count + 1

                # Print results
                # np.set_printoptions(precision=2)
                # print('Number of Samples = ', nSample)
                # print('Polynomial Degree = ', polyDegree)
                # print('Alpha = ', Alpha)
                # print('RMSE = ', rmse)
                # print('R2 = ', r2)
                # print('R2_test = ', r2_test)
                # print('Intercept = ', intercept)
                # print('Slope = ', slope, '\n\n')

                # Plot
    #             dpi = 100
    #             f = plt.figure(count, None, dpi)
    #             ax = f.add_subplot(111, projection='3d')
    #             ax.scatter(x, y, z, c='b', marker='*')
    #             ax.scatter(x, y, z_prediction, c='g', marker='o')
    #             ax.scatter(x_test, y_test, z_test, c='r', marker='.')
    #             plt.xlabel("x")
    #             plt.ylabel("y")
    #             plt.title("nSample = "+str(nSample)+", Poly Degree = "+str(polyDegree)+", Alpha = "+str(Alpha)+"\n"
    #                       +"R2 = "+str(r2)+", R2_test = "+str(r2_test))
    #             count = count + 1
    #
    # plt.show()

    df = pd.DataFrame(result)
    print(df)
    plt.figure()
    plt.plot(df['numSample'], df['R2_test'])
    plt.xlabel("numSample")
    plt.ylabel("R2_test")
    plt.show()




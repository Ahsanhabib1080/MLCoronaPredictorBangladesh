import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#### LOADING DATA ####
data = pd.read_csv('Covid19_data.csv', sep=',')
data = data[['Listofcases', 'total_cases']]
print('-' * 10);
print('Head');
print('-' * 10);
print(data.head())

#### PREPARING DATA ####
print('-' * 10);
print('PREPARING DATA');
print('-' * 10)
x = np.array(data['Listofcases']).reshape(-1, 1)
y = np.array(data['total_cases']).reshape(-1, 1)
plt.plot(y, '-r')

polyfeature = PolynomialFeatures(degree=7)
x = polyfeature.fit_transform(x)
# print(x)


#### TRAINING DATA ####
print('-' * 10);
print('TRAINING DATA');
print('-' * 10)
model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(x, y)
print(f'Accuracy: {round(accuracy * 100, 3)} %')
y0 = model.predict(x)

#### PREDICTION ####

#----PRINTING----
print('-' * 10);
print('PREDICTING DATA');
print('-' * 10)
print('-' * 10);
print("DATASET AVAILABLE TILL APRILL 5'th");
print('-' * 10)
#--------------

days = 1+int(input("How many days after you want to predict ? :")) # added 1+ for better Accurate Predictions
print(f'Prediction - Total Cases after {399 + days-1} days:', end='')
print(round(int(model.predict(polyfeature.fit_transform([[399 + days]]))), 2))
x1 = np.array(list(range(1, 399 + days))).reshape(-1, 1)
y1 = model.predict(polyfeature.fit_transform(x1))
plt.plot(y1, 'c')
plt.plot(y0, '--b')

plt.xlabel('Days')
plt.ylabel('Total Cases')
plt.title(f'Predicting Bangladesh CoronaVirus Total Cases after {399 + (days-1)} days')
plt.show()

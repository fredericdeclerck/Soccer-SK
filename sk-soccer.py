# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:43:38 2019

@author: fred
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:35:42 2019

@author: fred
"""

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#load soccer data
df = pd.read_excel('ECL2018-19.xlsx', sheet_name='Sheet1')

teams = df['Team']
possession = df['Possession%'].values.reshape(-1,1)
shotsPG = df['Shots pg'].values.reshape(-1,1)
goals = df['Goals']
AirWon  = df['AerialsWon'].values.reshape(-1,1)
Dribbles=df['Dribbles pg'].values.reshape(-1,1)
IntercPG = df['Interceptions pg'].values.reshape(-1,1)

# Use only one feature
## Split the data into training/testing sets
#possession_X_train, possession_X_test, shotsPG_y_train, shotsPG_y_test = train_test_split(possession, shotsPG, test_size=N, random_state=0)

#use multiple features
X = df[['Possession%','Dribbles pg','Pass%', 'Shots pg', 'Discipline', 'D Shots pg', 'Offsides pg']].values
y = df['goals pg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Create linear regression object
regressor = LinearRegression()  

regressor.fit(X_train, y_train)


coeff_df = pd.DataFrame(regressor.coef_)  


# Make predictions using the testing set
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# The coefficients
print('Coefficients: \n', coeff_df)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs

df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#plt.xticks(())
#plt.yticks(())

plt.show()
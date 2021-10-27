import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

concreteData = pd.read_csv("ConcreteData/Concrete_Data_Yeh.csv")

# Prediction target
# Concrete compressive strength MPa 
y = concreteData.csMPa

# Prediction Features
concreteFeatures = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']

X = concreteData[concreteFeatures]

# Train Data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define Model
concreteModel = DecisionTreeRegressor(random_state=1)

# Fit Model
concreteModel.fit(train_X,train_y)

predictedStrengths = concreteModel.predict(val_X)

print(mean_absolute_error(val_y, predictedStrengths))
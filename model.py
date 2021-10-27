import pandas as pd
from sklearn.tree import DecisionTreeRegressor

concreteData = pd.read_csv("ConcreteData/Concrete_Data_Yeh.csv")

# Prediction target
# Concrete compressive strength MPa 
y = concreteData.csMPa

# Prediction Features
concreteFeatures = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']

X = concreteData[concreteFeatures]

# Define Model
concreteModel = DecisionTreeRegressor(random_state=1)

concreteModel.fit(X,y)

print("Making prediction for 5 people")
print(X.head())
print("The predictions are:")
print(concreteModel.predict(X.head()))
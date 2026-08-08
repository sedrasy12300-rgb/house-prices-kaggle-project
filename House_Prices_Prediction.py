import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,GridSearchCV ,cross_val_score
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from sklearn.metrics import r2_score ,mean_absolute_error ,mean_squared_error ,root_mean_squared_error 
import matplotlib.pyplot as plt
import joblib


df =pd.read_csv("train.csv")

df["LotFrontage"]=df.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))
df["Alley"] = df["Alley"].fillna("None")
df["MasVnrType"] =df["MasVnrType"].fillna("None")
df["MasVnrArea"] =df.groupby("MasVnrType")['MasVnrArea'].transform(lambda x :x.fillna(x.median()))
df["BsmtQual"] =df["BsmtQual"].fillna("None")
df["BsmtCond"] =df["BsmtCond"].fillna("None")
df["BsmtExposure"] =df["BsmtExposure"].fillna("None")
df['BsmtFinType1'] =df["BsmtFinType1"].fillna("None")
df["BsmtFinType2"] =df["BsmtFinType2"].fillna("None")
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df["GarageType"] =df["GarageType"].fillna("None")
df["GarageYrBlt"] =df["GarageYrBlt"].fillna(0)
df["GarageFinish"] =df["GarageFinish"].fillna("None")
df["GarageQual"] = df["GarageQual"].fillna("None")
df["GarageCond"] =df["GarageCond"].fillna("None")
df["PoolQC"] = df["PoolQC"].fillna("None")
df["Fence"] =df["Fence"].fillna('None')
df["MiscFeature"] =df["MiscFeature"].fillna("None")

print(df.info())
df["TotalPorchSF"] =df["OpenPorchSF"] +df["WoodDeckSF"] +df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
df ["TotalSF"] =df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBathrooms"] = (df["FullBath"] + df["HalfBath"]*0.5 +df["BsmtFullBath"] +df["BsmtHalfBath"] *0.5)


df =df.drop('Id',axis=1)

Text_clounms =df.select_dtypes(include="object").columns
df =pd.get_dummies(df,columns=Text_clounms,drop_first=True)

x =df.drop("SalePrice",axis=1)
y =np.log1p(df["SalePrice"])

x_train, x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

train_columns =x_train.columns

model = RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print("R2S: ",r2_score(y_test,y_pred))
print("MSE: ",mean_squared_error(y_test,y_pred))
print("MAE: ",mean_absolute_error(y_test,y_pred))
print("RMSE: ",root_mean_squared_error(y_test,y_pred))

model_gradi =GradientBoostingRegressor(random_state=42)
model_gradi.fit(x_train,y_train)
y_gradi =model_gradi.predict(x_test)


print("R2S: ",r2_score(y_test,y_gradi))
print("MSE: ",mean_squared_error(y_test,y_gradi))
print("MAE: ",mean_absolute_error(y_test,y_gradi))
print("RMSE: ",root_mean_squared_error(y_test,y_gradi))


param ={"n_estimators": [100,200,300],"max_depth": [3,5,10,None],"min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4],"max_features":["sqrt","log2",None]}

grid =GridSearchCV(RandomForestRegressor(random_state=42),param_grid=param,cv=5,scoring="r2",n_jobs=-1)
grid.fit(x_train,y_train)

print("grid best estimator: ",grid.best_estimator_)
print("grid best scorw :",grid.best_score_)

model_best =grid.best_estimator_
joblib.dump(model_best,"model.pkl")
joblib.dump(train_columns,"columns.pkl")
y_grid =model_best.predict(x_test)

y_grid =np.expm1(y_grid)
y_test_real = np.expm1(y_test)

print("MAE: ",mean_absolute_error(y_test_real,y_grid))
print("MSE: ",mean_squared_error(y_test_real,y_grid))
print("RMSE: ",root_mean_squared_error(y_test_real,y_grid))
print("R2S :",r2_score(y_test_real,y_grid))

score =cross_val_score(model_best,x,y,cv=3,scoring= "r2")
print("score",score)
print("mean score",score.mean())
importance =pd.DataFrame({"Feature":x.columns,"Importance":model_best.feature_importances_})
importance =importance.sort_values(by="Importance",ascending=False)
print(importance.head(5))

importance.head(20).plot(x="Feature",y="Importance",kind="bar")
plt.show()


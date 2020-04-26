from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
##INSTRUCTIONS
#Download the .XLS file from https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
#Save it in the same folder as of python code
#Convert the .xls file to .csv and rename it 'covid19' and save it.

#Model 1
#-------------------------------------------------------------------------
series = read_csv('covid191.csv')
#Show Countries names
print('LIST OF COUNTRIES')
z=set(series.countriesAndTerritories)
con_list=list(z)
print(*con_list, sep="\n")

cont=[]
z=input('Enter country Name: ')
for i in range(1,len(series.cases)):
    if series.countriesAndTerritories[i]==z:
        cont.append(series.cases[i])

cont=np.flip(cont)
X = cont
size = int(len(X) * 0.90)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
print('Model 1 Observations')
print('--------------')
print('Test data predictions: ')
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#FORCAST
arr=np.arange(len(test)-1,len(test)+2)
fc, se, conf = model_fit.forecast(3, alpha=0.05)
fc_series = pd.Series(fc, index=arr)
if fc[1]==0:
    fc[1]=0

print('NEXT DAY PREDICTION(model1): ',fc[1])

#MODEL 2
#-------------------------------------------------------------------------    
#predict data
from sklearn.metrics import r2_score
d=np.arange(1,len(cont)+1)
x=d
y=cont

#-------------------------------------------------
#Optimization
order=7
mymodel = np.poly1d(np.polyfit(x,y,order))
r2=r2_score(y,mymodel(x))
zm=mymodel(len(cont)+2)
print('-------------------------------------------')
print('Model 2 Observations')
print('--------------')
print('R2 Score: ',r2*100)
if zm<0 or (abs(zm-fc[1])>500):
    zm=0
    print('Model 2 gives negative result')
print('NEXT DAY PREDICTION(model2)):',zm)
f_p=np.average([zm,fc[1]])
if f_p<0:
    f_p=0
print('--------------------------------------------')
print('FINAL PREDICTION: ',math.floor(f_p))



# plot
#---------------------------------------------------------------------------

#PLOT ALL FIGURES
fig=plt.figure()
plt1 = fig.add_subplot(221) 
plt2 = fig.add_subplot(224)
plt3 = fig.add_subplot(222) 
plt4 = fig.add_subplot(223)
xp=np.linspace(0,len(x)+2,len(x)*5)
plt1.plot(d,cont,'.',xp,mymodel(xp),'r')
plt.xlabel('Variation')
plt.ylabel('Number of Cases')
plt2.plot(test)
plt2.plot(predictions, color='red')
plt2.plot(fc_series,color='green')
plt3.plot(cont, 'r')
plt4.plot(cont,'.')
plt.xlabel('Variation')
plt.ylabel('Number of Cases')
plt1.set_title('Curve Fitting')
plt2.set_title('Prediction Using ARIMA')
plt3.set_title('Number of cases')
plt4.set_title('Scatter plot')
plt.show()
#----------------------------------------------------------------------------
#Completed Predicting Number of Cases
#----------------------------------------------------------------------------



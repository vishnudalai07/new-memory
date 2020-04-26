from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


#Read Data
data=pd.read_csv('covid19.csv')

#Show Countries names
z=set(data.countriesAndTerritories)
con_list=list(z)
print(*con_list, sep="\n")

#Enter the Country for analysis
country=input('Enter Countrys name from above list: ')
cont1=[]
day1=[]
mon1=[]
for i in range(1,len(data.day)):
    if data.countriesAndTerritories[i]==country:
        cont1.append(data.cases[i])
        day1.append(data.day[i])
        mon1.append(data.month[i])
cont=np.flip(cont1)
day=np.flip(day1)
month=np.flip(mon1)

#-------------------------------------------------
#Print Countries
from tabulate import tabulate as tb
for i in range(1,len(cont)):
    print(tb([[day[i],month[i],cont[i]]]))

#-------------------------------------------------    
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


print('R2 Score: ',r2*100)
if zm<0:
    zm=0
    print('Number of cases is negative from given Model')
print('Number of cases next day (predicted):',zm)
xp=np.linspace(0,len(x)+2,len(x)*5)
plt.plot(d,cont,'.',xp,mymodel(xp),'r')
plt.show()



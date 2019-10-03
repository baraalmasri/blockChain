   #importing library 
   import numpy as np 
   import matplotlib.pyplot as plt 
   import pandas as pd 
   from sklearn.model_selection import train_test_split 
   
   dataset = pd.read_csv('Salary_Data.csv') 
   #importing datasets 
   #here simpley we chouse the x colum and the other side we chouse the y colums 
   x= dataset.iloc[:,:-1].values 
   y= dataset.iloc[:,1].values 
   #splitting the dataset into training set and test set 
   x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1/3,random_state=0 ) 
   
   #fitting simple linear regression to the training set 
   #here we train the model simple linear regression model as we see 
   #fit the model by our data 
   #burada model ogrenmektedir 
   from sklearn.linear_model import LinearRegression 
   regressor= LinearRegression() 
   regressor.fit(x_train,y_train) 
   
   #predicting the test set results 
   #Test seti sonuçlarını tahmin etmek 
   #burada model ogrendikten sonra sonuclara bakacaz 
   """sadace x test setisi icin bakiyoruz burada yani 
   aslinda x_train yapmamiz gerek """ 
   y_predict= regressor.predict(x_test) 
   
   #visualising the training set results 
   '''burada  grafigi ciziyoruz x'ler ve 
   y'ler gercek datanin grafgi ''' 
   plt.scatter(x_train,y_train ,color='red') 
   '''burada  grafigi ciziyoruz x'ler gercek datanin x'leri 
   y'ler ise egitlimis  datanin grafgi ''' 
   plt.scatter(x_train,regressor.predict(x_train) ,color='blue') 
   plt.title('salary vs experience (training set)') 
   plt.xlabel('year of experience') 
   plt.ylabel('salary') 
   plt.show() 
   
   
   #test setisi icin grafigi cizecegiz 
   plt.scatter(x_test,y_test ,color='red') 
   '''burada  grafigi ciziyoruz x'ler gercek datanin x'leri 
   y'ler ise egitlimis  datanin grafgi ''' 
   plt.scatter(x_train,regressor.predict(x_train) ,color='blue') 
   plt.title('salary vs experience (training set)') 
   plt.xlabel('year of experience') 
   plt.ylabel('salary') 
   plt.show()

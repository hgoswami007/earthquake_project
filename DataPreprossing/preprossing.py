from sqlite3 import Time
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import pickle
from keras.models import load_model


class Preprossing:

    def __init__(self):
        pass

    def dataReading(self,csvpath):
        self.data = pd.read_csv(csvpath)
        data = self.data[['Date','Time','Latitude','Longitude','Depth','Magnitude']]
        return data

    def convertTimedata(self,data):
        timestamp = []
        for d, t in zip(data['Date'],data['Time']):
            try:
                ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
                timestamp.append(time.mktime(ts.timetuple()))
            except:
                timestamp.append('ValueError')
                 
                
        
        timeStamp = pd.Series(timestamp)
        
        data['Timestamp'] = timeStamp.values
        print(data['Timestamp'])
        final_data = data.drop(['Date','Time'],axis=1)
        final_data = final_data[final_data.Timestamp != 'ValueError']
        return final_data

    def split_data(self,data):
        X = data[['Timestamp','Latitude','Longitude']]
        y = data[['Magnitude','Depth']]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

        return X_train, X_test, y_train, y_test

    
    def saveModel(self,model,filename):
        # with open('D:\Earth\saveModel\\'+filename+'.pkl','wb') as f:
        #     pickle.dump(model,f)
        return model.save('D:\Earth\saveModel\\'+filename+".h5")
        
        

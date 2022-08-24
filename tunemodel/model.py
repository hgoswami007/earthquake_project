import pickle
from sklearn.model_selection import GridSearchCV
from DataPreprossing import preprossing
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


class model:
    def __init__(self,csvpath):
        self.preprossing = preprossing.Preprossing()
        self.data = self.preprossing.dataReading(csvpath)
        self.data = self.preprossing.convertTimedata(self.data)
        self.X_train,self.X_test,self.y_train,self.y_test = self.preprossing.split_data(self.data)

    def create_model(self,neurons,activation,optimizer,loss):
        
        model = Sequential()
        model.add(Dense(neurons,activation=activation,input_shape=(3,)))
        model.add(Dense(neurons,activation=activation))
        model.add(Dense(2,activation='softmax'))
        model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        
        # model.fit(self.X_train,self.y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(self.X_test,self.y_test))
        # [test_loss,test_acc] = model.evaluate(self.X_test,self.y_test)
        # return "Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc)
        return model
    
    def best_params(self):
        
        # data = self.preprossing.convertTimedata(self.data)
        # X_train, X_test, y_train, y_test = self.preprossing.split_data(data)
        model = KerasClassifier(build_fn=self.create_model)
        
        #Classify the neurons
        # neurons = [16]
        neurons = [16,64,128,256]
        #Classify the batch Size
        # batch_size = [10]
        batch_size = [10,20,50,100]
        #Classify the epochs
        epochs = [10]
        #Classify the activation functions
        activation = ['relu','sigmoid'] 
        # activation = ['relu','tanh','sigmoid','hard_sigmoid','linear','exponential']
        #Classify the optimizer
        optimizer = ['SGD','Adadelta'] 
        # optimizer = ['SGD','Adam','Adagrad','Adadelta','RMSprop','Adamax','Nadan']
        #Classify the loss
        loss = ['squared_hinge']

        self.params_grid = dict(neurons=neurons,batch_size=batch_size,epochs=epochs,
                            activation=activation,optimizer=optimizer,loss=loss)

    
        self.grid = GridSearchCV(estimator=model,param_grid=self.params_grid)
        
        self.X_train = self.X_train.values.astype(np.float32)
        self.y_train = self.y_train.values.astype(np.float32)
        
        
        self.grid_result = self.grid.fit(self.X_train,self.y_train)
        self.grid_result.best_params_
        # print(best_params)
        return self.grid_result.best_params_

    def modelwithBestParams(self):
        self.params = self.best_params()
        model = Sequential()
        self.X_train = np.array(self.X_train).astype(np.float32)
        self.y_train = np.array(self.y_train).astype(np.float32)
        self.X_test = np.array(self.X_test).astype(np.float32)
        self.y_test = np.array(self.y_test).astype(np.float32)


        model.add(Dense(units=(self.params['neurons']),activation=self.params['activation'],input_shape=(3,)))
        model.add(Dense(units=self.params['neurons'],activation=self.params['activation']))
        model.add(Dense(2,activation='softmax'))


        model.compile(optimizer=self.params['optimizer'], loss=self.params['loss'],metrics=['accuracy'])
        
        model.fit(self.X_train, self.y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], verbose=1, validation_data=(self.X_test,self.y_test))
        
        self.preprossing.saveModel(model,'earthquake')
        # model.save('earthquake.h5')
        [test_loss,test_acc] = model.evaluate(self.X_test,self.y_test)
        return "Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc)
    
    def loadmodel(self,filename):
        # with open('D:\Earth\saveModel\\'+ filename + '.pkl','rb') as f:
        #     model = pickle.load(f)
            self.X_test = np.array(self.X_test).astype(np.float32)
            self.y_test = np.array(self.y_test).astype(np.float32)
            loaded_model = load_model('D:\Earth\saveModel\\'+filename+'.h5')
            test_loss,test_acc = loaded_model.evaluate(self.X_test,self.y_test)
            print('accuracy====>',test_acc,test_loss)
            return test_acc,test_loss

     
    

        



            

    
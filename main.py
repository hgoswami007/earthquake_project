from flask import Flask,render_template,request
from flask_cors import cross_origin
from tunemodel import model
from DataPreprossing import preprossing

app = Flask(__name__)

@app.route('/',methods=['get'])
@cross_origin()
def homepage():
    tune_object = model.model(csvpath='D:\Earth\Dataset\database.csv')
    # tune_object.best_params()
    test_acc,test_loss = tune_object.loadmodel('earthquake')
    
    return render_template('home.html',prediction_text=f"Model accuracy is : {test_acc} and loss is {test_loss}")
    

if (__name__) == '__main__':
    app.run()
    
   
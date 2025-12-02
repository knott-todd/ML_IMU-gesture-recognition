from flask import Flask, request
from flask_restful import Api, Resource, reqparse
# !pip install -U flask-cors
from flask_cors import CORS, cross_origin

app = Flask(__name__)
api = Api(app)
CORS(app)


config_get_args = reqparse.RequestParser()
config_get_args.add_argument("num_readings_per_sample_per_mpu", type=int, help="num_readings_per_sample_per_mpu is required", required=True)
config_get_args.add_argument("num_mpus", type=int, help="num_mpus is required", required=True)
config_get_args.add_argument("num_inputs", type=int)

class Config(Resource):
    def put(self):
        args = config_get_args.parse_args()
        init_config(args['num_readings_per_sample_per_mpu'], args['num_mpus'], args['num_inputs'])
        return args
    
api.add_resource(Config, "/config")


add_get_args = reqparse.RequestParser()
add_get_args.add_argument("path", type=str, help="Path to CSV data is required", required=True)

class AddGesture(Resource):
    def put(self):
        args = add_get_args.parse_args()
        add_gesture(args['path'])
        return args
        
api.add_resource(AddGesture, "/add")

class Build(Resource):
    def get(self):
        model = build_model()
        return model
    
api.add_resource(Build, "/build")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)



def init_config(_num_readings_per_sample_per_mpu, _num_mpus, _num_inputs):
    global pd
    import pandas as pd
    global train_test_split
    from sklearn.model_selection import train_test_split
    global glob
    import glob
    global os
    import os
    global numpy
    import numpy

    global num_readings_per_sample_per_mpu 
    num_readings_per_sample_per_mpu = _num_readings_per_sample_per_mpu
    global num_mpus 
    num_mpus = _num_mpus
    global num_inputs
    num_inputs = 6
    
    if _num_inputs != None:
        num_inputs = _num_inputs
        
    global gesture_index
    gesture_index = 0
    global X
    X = None
    global y
    y = None


def add_gesture(gesture_csv_path):
    global gesture_index
    global num_readings_per_sample_per_mpu
    global num_mpus
    global num_inputs
    print("Gesture index: ", gesture_index)

#     gesture_csv_path = r'C:\Users\olusa\Documents\VS Code\Node\MVP\ges-rec-api\gesture_recordings\Bap\Bap_parsed.csv' # HC
    gesture_csv = pd.read_csv(gesture_csv_path)

    num_readings = len(gesture_csv)
    num_samples = int(num_readings / num_readings_per_sample_per_mpu / num_mpus)

    gesture_X = numpy.zeros((num_samples,num_readings_per_sample_per_mpu,num_inputs * num_mpus)) # num samples x num readings per sample x num inputs

    temp_X = numpy.zeros((num_mpus, int(num_readings / num_mpus), num_inputs))

    for mpu_index in range(num_mpus):
        for reading_index in range(num_readings):
            if (reading_index-mpu_index) % num_mpus == 0:
                temp_X[mpu_index][int(reading_index/num_mpus)] = gesture_csv.iloc[reading_index, 1:]

    for sample_index in range(num_samples):
        for mpu_index in range(num_mpus):
            for reading_index in range(num_readings_per_sample_per_mpu):            
                gesture_X[sample_index][reading_index][num_inputs*mpu_index:num_inputs*(mpu_index+1)] = temp_X[mpu_index][sample_index*num_readings_per_sample_per_mpu+reading_index]


    gesture_y = numpy.full(num_samples, gesture_index) # Creating y vector with gesture index

    global X
    global y
    
    if gesture_index == 0:
        X = gesture_X
        y = gesture_y
    else:
        X = numpy.concatenate([X, gesture_X])
        y = numpy.concatenate([y, gesture_y])
        
    print("oi")
    print(gesture_y)
    print(num_samples)


    gesture_index += 1



def build_model():
    global X
    global y
    
    print(X)
    print(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras import regularizers
    from sklearn.metrics import accuracy_score
    
    model = Sequential()
    
    global num_readings_per_sample_per_mpu
    global num_inputs
    global num_mpus
    global gesture_index

    model.add(Flatten(input_shape=(num_readings_per_sample_per_mpu, num_inputs * num_mpus)))
    model.add(Dense(units=gesture_index, activation='softmax', kernel_regularizer=regularizers.l2(0.1)))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=40, batch_size=16, validation_split=0.2)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    import tensorflow as tf
    print(tf.__version__)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter.convert()
    
    model.save('model')
    
    open("converted_model.tflite", "wb").write(tflite_quant_model)
    
    from tinymlgen import port

    c_code = port(model, pretty_print=True, variable_name = "model")
    return c_code
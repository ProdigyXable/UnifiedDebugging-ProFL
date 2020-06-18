from sklearn.model_selection import KFold
import tensorflow
import numpy
import os
import sys

# --- Machine Learning Hyper Parameters [START] --- #
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 10000
CLASSES = 2 # Buggy vs Non-Buggy
LAYER_NEURONS = 128
FEATURE_NUM = 124 # 34 + (15 * 6)
DROPOUT_RATE = 0.3
# --- Machine Learning Hyper Parameters [END] --- #

# --- OTHER [START] --- #
# 278 buggy methods in dataset (excluding Math-66)
# --- OTHER [END] --- #

def processNumFeats(data):
    result = []

    for line in data:
        line = line.strip()
        line_data = line.split(",")[1:]
        row_data = []
        for item in line_data:
            if(item == 10000000):
                item = 1000
            row_data.append(float(item))

        result.append(row_data)
    return result

def readNumFeats(dir, project):
    data = {}
    for f in os.listdir(dir):
        #if f.endswith(".numFeats") and not "Math-66" in f and project in f:
        if f.endswith(".numFeats") and not "Math-66" in f:

            line_data = open(os.path.join(dir, f),"r").readlines()[1:]
            processed_data = processNumFeats(line_data)
            
            id = f.split(".")[0]
            data[id] = []
            data[id].extend(numpy.array(processed_data))
            
    return data

def processClassFeats(data):
    result_data = []
    result_sig = []

    for line in data:
        line = line.strip().split(",")
        methodSig = line[0]
        line_data = line[1:]
        
        row_data = []
        for item in line_data:
            row_data.append(int(item))

        result_data.append(row_data)
        result_sig.append(methodSig)
    return (result_data, result_sig)

def readClassFeats(dir, project):
    data = {}
    data_sigs = {}
    for f in os.listdir(dir):

        #if f.endswith(".classFeats") and not "Math-66" in f and project in f:
        if f.endswith(".classFeats") and not "Math-66" in f:
            (processed_data, sigs) = processClassFeats(open(os.path.join(dir, f),"r").readlines()[1:])

            id = f.split(".")[0]
            data[id] = []
            data_sigs[id] = []

            data[id].extend(numpy.array(processed_data))
            data_sigs[id].extend(sigs)
    return (data, data_sigs)

def makeNeuralNetwork():
    # activation='relu', kernel_regularizer='l1_l2', activity_regularizer='l1_l2', bias_regularizer='l1_l2'

    model = tensorflow.keras.models.Sequential()
    # --- HIDDEN LAYER 1 --- #
    model.add(tensorflow.keras.layers.Dense(LAYER_NEURONS, kernel_regularizer='l2', input_shape=(FEATURE_NUM,)))
    model.add(tensorflow.keras.layers.LeakyReLU())
    model.add(tensorflow.keras.layers.Dropout(DROPOUT_RATE))
    
    # --- HIDDEN LAYER 2 --- #
    model.add(tensorflow.keras.layers.Dense(LAYER_NEURONS / 2, kernel_regularizer='l2'))
    model.add(tensorflow.keras.layers.LeakyReLU())
    model.add(tensorflow.keras.layers.Dropout(DROPOUT_RATE))
    
    # --- OUTPUT LAYER --- #
    model.add(tensorflow.keras.layers.Dense(CLASSES,kernel_regularizer='l2', activation='sigmoid'))
    #model.add(tensorflow.keras.layers.Softmax())

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.Adagrad(), 
                  metrics=[
                      'accuracy'
                      ])

    return model

def model():
    # --- DATA PROCESSING --- #
    print("Executing data preprocessing")

    print("Importing raw data")
    projects = ["Chart", "Time", "Lang", "Math"]
    for project in projects:

        raw_output_train, sigs = readClassFeats(sys.argv[1], project)
        raw_input_train = readNumFeats(sys.argv[1], project)

        for project_prediction_id in raw_input_train.keys():
            print("Processing", project_prediction_id)

            # --- Create Neural Network --- #
            network_model = makeNeuralNetwork()

            # --- Create Training / Test Data --- #
            training_data_input = []
            training_data_output = []

            for other_project in raw_input_train.keys():
                if not project_prediction_id is other_project:
                    training_data_input.extend(raw_input_train[other_project])
                    training_data_output.extend(raw_output_train[other_project])

            training_data_input = tensorflow.keras.utils.normalize(numpy.array(training_data_input))
            training_data_output = numpy.array(training_data_output)

            class_weights = {}
            class_weights[0] = 1
            class_weights[1] = setClassWeight(training_data_output)

            print("Class weights to set to", class_weights)
        
            training_data_output = tensorflow.keras.utils.to_categorical(training_data_output, CLASSES)
        
            test_data_input = tensorflow.keras.utils.normalize(numpy.array(raw_input_train[project_prediction_id]))
            test_data_output = numpy.array(raw_output_train[project_prediction_id])

            validation_data_splitter = KFold(n_splits=4)

            print("--- Training Results ---")
            for indicies_train, indicies_validate in validation_data_splitter.split(training_data_input, y=training_data_output):   
                network_model.fit(class_weight=class_weights,
                                  x=training_data_input[indicies_train], y=training_data_output[indicies_train],
                                  batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, 
                                  validation_data=(training_data_input[indicies_validate], training_data_output[indicies_validate]),
                                  callbacks=getCallback(),
                                  )
        
            print("--- Prediction Results ---")

            #output_file = open("profl-feats-only_" + project_prediction_id + ".susValues", "w")
            output_file = open(project_prediction_id + ".susValues", "w")
            for k, (i) in enumerate(network_model.predict(test_data_input, verbose=1)):
               prediction = numpy.argmax(i)
               print('\t-', i, "->" , prediction, prediction == test_data_output[k], sigs[project_prediction_id][k])
               output_file.write(",".join([sigs[project_prediction_id][k], str(i[prediction])]))
               output_file.write("\n")

        
            print("\n", "----------------", "\n")
            output_file.close()

def setClassWeight(data):

    other = 0
    positive = 0

    for row in data:
        if(row == 1):
            positive += 1
        else:
            other += 1

    weight = other / positive
    return weight

def getCallback():
    return [
            tensorflow.keras.callbacks.TerminateOnNaN(),
            tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=EPOCHS / 50, restore_best_weights=False),
            tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCHS / 100, restore_best_weights=False, min_delta=0.0001)
            ]

model()
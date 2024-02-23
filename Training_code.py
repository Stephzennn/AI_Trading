
from All_common_functions import *

# Define the name of the coin
coin_names = ("ETCUSDT")
coin = 'ETCUSDT'

# Define the dates for training
dateoftraining = ["26 NOV, 2020", "26 NOV , 2022"]

# Load the training data for the coin from a numpy file
datatrain = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatrain.npy')

# Print the shape of the training data
print(datatrain.shape)

# Load the training data for Bitcoin (BTC) from a numpy file
datatrainbtc = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatrainbtc.npy')

# Load the test data for the coin from a numpy file
datatest1 = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest1.npy')

# Load the test data for Bitcoin (BTC) from a numpy file
datatest1btc = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest1btc.npy')

# Load more test data for the coin from a numpy file
datatest2 = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest2.npy')

# Load more test data for Bitcoin (BTC) from a numpy file
datatest2btc = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest2btc.npy')

# Load even more test data for the coin from a numpy file
datatest3 = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest3.npy')

# Load even more test data for Bitcoin (BTC) from a numpy file
datatest3btc = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest3btc.npy')

# Load the final set of test data for the coin from a numpy file
datatest4 = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest4.npy')

# Load the final set of test data for Bitcoin (BTC) from a numpy file
datatest4btc = np.load('/content/drive/MyDrive/Recurrent data. To be fed to Colab /'+coin[0:-4]+'datatest4btc.npy')
# Define a function to calculate the percentage movement
def percentagemovement(open, close):
    open = round(float(open), 3)
    close = round(float(close), 3)
    pm = (abs(open - close) / open) * 100
    return round(pm, 2)

def run_model(array):
    [[coin]] = array[0:1,0:1]
    [[modelname]] = array[0:1,1:2]
    [[modelname1]] = array[0:1, 2:3]
    [[modelname2]] = array[0:1, 3:4]
    [[modelname3]] = array[0:1, 4:5]
    c_four_hour = datatrain
    c_four_hourBTC = datatrainbtc
    c_four_hour_TEST = datatest1
    c_four_hour_TESTBTC = datatest1btc
    
    

    X_train, y_train,n_steps,n_features,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11,sc12,sc13,sc14,sc15,sc16,sc17,sc18,sc19,sc20,sc21,sc22,sc23,sc24,sc25,sc26,sc27,sc28,sc29,sc30,sc31,sc32,sc33,sc34,sc35,sc36,sc37,sc38,sc39,crnew = prepare(c_four_hour,c_four_hourBTC)
    print(X_train.shape)

    X_test, y_test,new1 , mayeb1,crnew1 = prepare_test(c_four_hour_TEST,c_four_hour_TESTBTC,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11,sc12,sc13,sc14,sc15,sc16,sc17,sc18,sc19,sc20,sc21,sc22,sc23,sc24,sc25,sc26,sc27,sc28,sc29,sc30,sc31,sc32,sc33,sc34,sc35,sc36,sc37,sc38,sc39)
    print(X_test.shape)
    c_four_hour_TEST1 = datatest2

    c_four_hour_TESTBTC1 = datatest2btc

    X_test1, y_test1, new11, mayeb11, crnew11 = prepare_test(c_four_hour_TEST1, c_four_hour_TESTBTC1, sc1, sc2, sc3,
                                                                sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                sc35, sc36, sc37, sc38, sc39)
    c_four_hour_TEST2 = datatest3
    c_four_hour_TESTBTC2 = datatest3btc

    X_test2, y_test2, new12, mayeb12, crnew12 = prepare_test(c_four_hour_TEST2, c_four_hour_TESTBTC2, sc1, sc2, sc3,
                                                                sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                sc35, sc36, sc37, sc38, sc39)
    c_four_hour_TEST3 = datatest4

    c_four_hour_TESTBTC3 = datatest4btc

    X_test3, y_test3, new13, mayeb13, crnew13 = prepare_test(c_four_hour_TEST3, c_four_hour_TESTBTC3, sc1, sc2, sc3,
                                                                sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                sc35, sc36, sc37, sc38, sc39)

    # Set a scaling factor for the number of units in each layer
    crar = 1.9
    one = int(crar * 100)
    two = int(100 * crar)
    three = int(100 * crar)
    four = int(60 * crar)

    # Import necessary layers
    from keras.layers import BatchNormalization
    from keras.layers import Bidirectional

    # Initialize a Sequential model
    model = Sequential()

    # Add the first LSTM layer with 'one' units, return sequences for next LSTM layer, tanh activation function, and specify input shape
    model.add((LSTM((one), return_sequences=True, activation='tanh', input_shape=(n_steps, n_features))))

    # Add a Dropout layer to prevent overfitting
    model.add(Dropout(0.2))

    # Add a Batch Normalization layer to standardize the inputs to the next layer
    model.add(BatchNormalization())

    # Add the second LSTM layer with 'two' units and return sequences for next LSTM layer
    model.add((LSTM(units=(two), return_sequences=True, activation='tanh')))

    # Add another Dropout layer
    model.add(Dropout(0.2))

    # Add another Batch Normalization layer
    model.add(BatchNormalization())

    # Add the third LSTM layer with 'three' units and return sequences for next LSTM layer
    model.add((LSTM(units=(three), return_sequences=True, activation='tanh')))

    # Add another Dropout layer
    model.add(Dropout(0.2))

    # Add another Batch Normalization layer
    model.add(BatchNormalization())

    # Add the fourth LSTM layer with 'four' units
    model.add((LSTM(units=(four), activation='tanh')))

    # Add another Dropout layer
    model.add(Dropout(0.2))

    # Add another Batch Normalization layer
    model.add(BatchNormalization())

    # Add a Dense layer with 32 units and ReLU activation function
    model.add(Dense(32, activation="relu"))

    # Add another Dropout layer
    model.add(Dropout(0.2))

    # Add the output Dense layer with 1 unit and sigmoid activation function for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Define the optimizer with learning rate 0.001 and decay 1e-6
    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile the model with binary crossentropy loss function for binary classification and accuracy as the metric
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    # Initialize the best results for each test set
    result1 = 0
    result2 = 0
    result3 = 0
    result4 = 0

    # Initialize counters
    count = 0
    passs = 0

    # Train the model for a certain number of epochs
    while count <= 70:
        # Some operation related to the count variable
        too = count
        too1 = str((too / 10) * 1000)
        too2 = too1[1:]
        too3 = float(too2)
        if too3 <= 0:
            print(too)

        # Fit the model on the training data
        model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0)

        # Evaluate the model on each test set and get the profit
        presult1 = find_finalpo(model, X_test, c_four_hour_TEST, y_test)
        presult2 = find_finalpo(model, X_test1, c_four_hour_TEST1, y_test1)
        presult3 = find_finalpo(model, X_test2, c_four_hour_TEST2, y_test2)
        presult4 = find_finalpo(model, X_test3, c_four_hour_TEST3, y_test3)

        # If the profit is better than the best result so far, save the model and update the best result
        if (presult1 >= result1):
            result1 = presult1
            model.save(modelname)
        if (presult2 >= result2):
            result2 = presult2
            model.save(modelname1)
        if (presult3 >= result3):
            result3 = presult3
            model.save(modelname2)
        if (presult4 >= result4):
            result4 = presult4
            model.save(modelname3)

        # Increment the counter
        count += 1

        # Clean up to free memory
        del (presult1)
        del (presult2)
        del (presult3)
        del (presult4)
        gc.collect()
        k.backend.clear_session()

    # Print the best results for each test set
    print(result1)
    print(result2)
    print(result3)
    print(result4)
    return






def create_model_names(coin_name, dateoftraining,datatrain,datatrainbtc,datatest1,datatest1btc,datatest2,datatest2btc,datatest3,datatest3btc,datatest4,datatest4btc):
    
    # Initialize an empty list 'lisst'
    lisst = []

    # Initialize an empty list 'list2'
    list2 = []

    # Append the coin name to 'lisst'
    lisst.append(coin_name)

    # Append the result of 'extract_four_names' to 'list2'
    list2.append(extract_four_names(coin_name))

    # Convert 'lisst' to a numpy array and reshape it to have one column
    lisst = np.array(lisst)
    lisst = np.reshape(lisst, (lisst.shape[0], 1))

    # Convert 'list2' to a numpy array
    list2 = np.array(list2)

    # Concatenate 'lisst' and 'list2' horizontally
    lisst3 = hstack((lisst, list2))
    def run_model(array):
        [[coin]] = array[0:1,0:1]
        [[modelname]] = array[0:1,1:2]
        [[modelname1]] = array[0:1, 2:3]
        [[modelname2]] = array[0:1, 3:4]
        [[modelname3]] = array[0:1, 4:5]
        c_four_hour = datatrain
        c_four_hourBTC = datatrainbtc
        c_four_hour_TEST = datatest1
        c_four_hour_TESTBTC = datatest1btc
        
        

        X_train, y_train,n_steps,n_features,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11,sc12,sc13,sc14,sc15,sc16,sc17,sc18,sc19,sc20,sc21,sc22,sc23,sc24,sc25,sc26,sc27,sc28,sc29,sc30,sc31,sc32,sc33,sc34,sc35,sc36,sc37,sc38,sc39,crnew = prepare(c_four_hour,c_four_hourBTC)
        print(X_train.shape)

        X_test, y_test,new1 , mayeb1,crnew1 = prepare_test(c_four_hour_TEST,c_four_hour_TESTBTC,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11,sc12,sc13,sc14,sc15,sc16,sc17,sc18,sc19,sc20,sc21,sc22,sc23,sc24,sc25,sc26,sc27,sc28,sc29,sc30,sc31,sc32,sc33,sc34,sc35,sc36,sc37,sc38,sc39)
        print(X_test.shape)
        c_four_hour_TEST1 = datatest2

        c_four_hour_TESTBTC1 = datatest2btc

        X_test1, y_test1, new11, mayeb11, crnew11 = prepare_test(c_four_hour_TEST1, c_four_hour_TESTBTC1, sc1, sc2, sc3,
                                                                 sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                 sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                 sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                 sc35, sc36, sc37, sc38, sc39)
        c_four_hour_TEST2 = datatest3
        c_four_hour_TESTBTC2 = datatest3btc

        X_test2, y_test2, new12, mayeb12, crnew12 = prepare_test(c_four_hour_TEST2, c_four_hour_TESTBTC2, sc1, sc2, sc3,
                                                                 sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                 sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                 sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                 sc35, sc36, sc37, sc38, sc39)
        c_four_hour_TEST3 = datatest4

        c_four_hour_TESTBTC3 = datatest4btc

        X_test3, y_test3, new13, mayeb13, crnew13 = prepare_test(c_four_hour_TEST3, c_four_hour_TESTBTC3, sc1, sc2, sc3,
                                                                 sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14,
                                                                 sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24,
                                                                 sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34,
                                                                 sc35, sc36, sc37, sc38, sc39)

        # Set a scaling factor for the number of units in each layer
        crar = 1.9
        one = int(crar * 100)
        two = int(100 * crar)
        three = int(100 * crar)
        four = int(60 * crar)

        # Import necessary layers
        from keras.layers import BatchNormalization
        from keras.layers import Bidirectional

        # Initialize a Sequential model
        model = Sequential()

        # Add the first LSTM layer with 'one' units, return sequences for next LSTM layer, tanh activation function, and specify input shape
        model.add((LSTM((one), return_sequences=True, activation='tanh', input_shape=(n_steps, n_features))))

        # Add a Dropout layer to prevent overfitting
        model.add(Dropout(0.2))

        # Add a Batch Normalization layer to standardize the inputs to the next layer
        model.add(BatchNormalization())

        # Add the second LSTM layer with 'two' units and return sequences for next LSTM layer
        model.add((LSTM(units=(two), return_sequences=True, activation='tanh')))

        # Add another Dropout layer
        model.add(Dropout(0.2))

        # Add another Batch Normalization layer
        model.add(BatchNormalization())

        # Add the third LSTM layer with 'three' units and return sequences for next LSTM layer
        model.add((LSTM(units=(three), return_sequences=True, activation='tanh')))

        # Add another Dropout layer
        model.add(Dropout(0.2))

        # Add another Batch Normalization layer
        model.add(BatchNormalization())

        # Add the fourth LSTM layer with 'four' units
        model.add((LSTM(units=(four), activation='tanh')))

        # Add another Dropout layer
        model.add(Dropout(0.2))

        # Add another Batch Normalization layer
        model.add(BatchNormalization())

        # Add a Dense layer with 32 units and ReLU activation function
        model.add(Dense(32, activation="relu"))

        # Add another Dropout layer
        model.add(Dropout(0.2))

        # Add the output Dense layer with 1 unit and sigmoid activation function for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Define the optimizer with learning rate 0.001 and decay 1e-6
        optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

        # Compile the model with binary crossentropy loss function for binary classification and accuracy as the metric
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        def find_finalpo(model, xtest, ct, y_test):
            # Predict the output using the model
            predict = model.predict(xtest, verbose=0)

            # Convert the predictions to binary values based on a threshold of 0.5
            pred3 = np.array([0 if x <= 0.5 else 1 for x in predict]).reshape(-1, 1)

            # Calculate the number of changes in prediction
            sume = sum(x != y for x, y in zip(pred3[:-1], pred3[1:]))

            # Calculate the commission based on the number of changes
            com = sume * 0.07

            

            # Calculate the percentage movement for each data point
            per = np.array([percentagemovement(x, y) for x, y in zip(ct[:, 1:2], ct[:, 4:5])]).reshape(-1, 1)

            # Get the percentage movement for the test data
            pern = per[290:, :]

            # Calculate the total profit or loss
            collu = sum(z if float(x) == float(y) else -z for x, y, z in zip(pred3, y_test, pern))

            # Subtract the commission to get the final profit or loss
            finalpo = collu - com

            return finalpo

        # Initialize the best results for each test set
        result1 = 0
        result2 = 0
        result3 = 0
        result4 = 0

        # Initialize counters
        count = 0
        passs = 0

        # Train the model for a certain number of epochs
        while count <= 70:
            # Some operation related to the count variable
            too = count
            too1 = str((too / 10) * 1000)
            too2 = too1[1:]
            too3 = float(too2)
            if too3 <= 0:
                print(too)

            # Fit the model on the training data
            model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0)

            # Evaluate the model on each test set and get the profit
            presult1 = find_finalpo(model, X_test, c_four_hour_TEST, y_test)
            presult2 = find_finalpo(model, X_test1, c_four_hour_TEST1, y_test1)
            presult3 = find_finalpo(model, X_test2, c_four_hour_TEST2, y_test2)
            presult4 = find_finalpo(model, X_test3, c_four_hour_TEST3, y_test3)

            # If the profit is better than the best result so far, save the model and update the best result
            if (presult1 >= result1):
                result1 = presult1
                model.save(modelname)
            if (presult2 >= result2):
                result2 = presult2
                model.save(modelname1)
            if (presult3 >= result3):
                result3 = presult3
                model.save(modelname2)
            if (presult4 >= result4):
                result4 = presult4
                model.save(modelname3)

            # Increment the counter
            count += 1

            # Clean up to free memory
            del (presult1)
            del (presult2)
            del (presult3)
            del (presult4)
            gc.collect()
            k.backend.clear_session()

        # Print the best results for each test set
        print(result1)
        print(result2)
        print(result3)
        print(result4)
        return
    run_model(lisst3)

create_model_names(coin_names, dateoftraining,datatrain,datatrainbtc,datatest1,datatest1btc,datatest2,datatest2btc,datatest3,datatest3btc,datatest4,datatest4btc)


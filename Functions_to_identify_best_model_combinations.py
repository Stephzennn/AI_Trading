
from All_common_functions import *

def download_data(coin, date):
    # Print '1' to the console for debugging or progress tracking
    print('1')
    
    # Fetch historical data for the specified coin at 30-minute intervals between the specified dates
    # The data is fetched from the futures market
    c_four_hour = client.get_historical_klines(coin, Client.KLINE_INTERVAL_30MINUTE, date[0], date[1],
                                               klines_type=HistoricalKlinesType.FUTURES)
    # Convert the fetched data into a numpy array
    c_four_hour = np.array(c_four_hour)

    # Fetch historical data for Bitcoin (BTC) at 30-minute intervals between the specified dates
    # The data is fetched from the futures market
    c_four_hourBTC = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, date[0],
                                                  date[1], klines_type=HistoricalKlinesType.FUTURES)
    # Convert the fetched data into a numpy array
    c_four_hourBTC = np.array(c_four_hourBTC)
    
    # Return the fetched data for the specified coin and for Bitcoin
    return c_four_hour, c_four_hourBTC


coin1 = 'ETCUSDT'
coin2 = 'ALPHAUSDT'
coin3 = 'SUSHIUSDT'
coinlist = [coin1,coin2,coin3]


# This function takes a date string in the format "YYYY-MM-DD-YYYY-MM-DD" and converts it into two separate date strings in the format "DD MM, YYYY". The function returns these two date strings in a list.
def convert_test_date(date):
    # The input date string
    firsttime = date
    
    # Find the positions of the hyphens that separate the year, month, and day
    startyear = firsttime.find('-')
    startmonth = firsttime.find('-', (startyear + 1))
    startday = firsttime.find('-', (startmonth + 1))
    endyear = firsttime.find('-', (startday + 1))
    endmonth = firsttime.find('-', (endyear + 1))
    endday = firsttime.find('-', (endmonth + 1))
    
    # Extract the year, month, and day from the first date in the input string
    y = firsttime[(startyear + 1):(startyear + 5)]
    m = firsttime[(startmonth + 1):(startmonth + 3)]
    d = firsttime[(startday + 1):(startday + 3)]
    
    # Extract the year, month, and day from the second date in the input string
    y2 = firsttime[(endyear + 1):(endyear + 5)]
    m2 = firsttime[(endmonth + 1):(endmonth + 3)]
    d2 = firsttime[(endday + 1):(endday + 3)]
    
    # Create the new date strings in the format "DD MM, YYYY"
    string1 = d + ' ' + m + ',' + ' ' + y
    string2 = d2 + ' ' + m2 + ',' + ' ' + y2
    
    # Return the two new date strings in a list
    daate = [string1, string2]
    return daate


four = forloop(0, 4, 'ethusdt')
Date  = convert_test_date(four)


# This function takes a model name as input and extracts four names from it using the 'extract_four_names' function. 
# It returns these four names in a list.
def extract_four_namess(modelname):
    # Use the 'extract_four_names' function to extract four names from the model name
    a, s, d, f = extract_four_names(modelname)
    
    # Create a list of the extracted names
    lisst = [a, s, d, f]
    
    # Return the list of names
    return lisst

d = []
for x in coinlist:
    d.append(extract_four_namess(x))
d = np.array(d)

[[model1]] = d[0:1,0:1] 
[[model1point1]] = d[0:1,1:2] 
[[model1point2]] = d[0:1,2:3]  
[[model1point3]] = d[0:1,3:4] 
[[model2]] = d[1:2,0:1] 
[[model2point1]] = d[1:2,1:2]  
[[model2point2]] = d[1:2,2:3] 
[[model2point3]] = d[1:2,3:4]  
[[model3]] = d[2:3,0:1]
[[model3point1]] = d[2:3,1:2]  
[[model3point2]] = d[2:3,2:3]  
[[model3point3]] = d[2:3,3:4]  


e = load_model(model3point3)


dateoftraining = ["26 NOV, 2020", "26 NOV , 2022"]

coindata = []
coindatabtc = []
for o in coinlist:
    ch, cb = download_data(o,dateoftraining)
    coindata.append(ch)
    coindatabtc.append(cb)
coinnumber1 = coinlist[0]
ch1,cb1 = download_data(coinnumber1,dateoftraining)
modeldata1 = ch1
model1point1_data = ch1
model1point2_data = ch1
model1point3_data = ch1
modeldata1btc = cb1
model1point1btc_data = cb1
model1point2btc_data = cb1
model1point3btc_data = cb1
coinnumber2 = coinlist[1]
ch2,cb2 = download_data(coinnumber2,dateoftraining)
modeldata2 = ch2
model2point1_data = ch2
model2point2_data = ch2
model2point3_data = ch2
modeldata2btc = cb2
model2point1btc_data = cb2
model2point2btc_data = cb2
model2point3btc_data = cb2
coinnumber3 = coinlist[2]
ch3,cb3 = download_data(coinnumber3,dateoftraining)
modeldata3 = ch3
model3point1_data = ch3
model3point2_data = ch3
model3point3_data = ch3
modeldata3btc = cb3
model3point1btc_data = cb3
model3point2btc_data = cb3
model3point3btc_data = cb3



# This function is a decorator that takes a function 'find_finalpo' as an argument. 
# The decorator function defines an inner function that loads a model and then calls the 'find_finalpo' function with the loaded model and other arguments.
def Wrap_find_finalpo(find_finalpo):
            def inner(model, xtest, ct, y_test):
                model = load_model(model)
                ans = find_finalpo(model, xtest, ct, y_test)
                return ans
            return inner


find_finalpo = Wrap_find_finalpo(find_finalpo)



# This function takes a list of models, a list of coins, and a date as input. 
# It uses the models to make predictions for each coin on the given date. 
# The function returns a DataFrame where each row corresponds to a coin and each column corresponds to a model. 
# The entries in the DataFrame are the predictions made by the models for the coins.
def frommodellist(modellist, coinlist, date):
    # Convert the list of models to a numpy array and get its shape
    w = np.array(modellist)
    w = w.shape[0]
    
    # Get the length of the coinlist
    l = len(coinlist)
    
    # Initialize a list of zeros with the same length as the number of models
    wlist = ['0'] * w
    wlist = np.array(wlist)
    
    # Stack the list of zeros vertically for each coin
    we = np.vstack([wlist]*l)
    
    # Create an array of coin names with 'MODEL' at the start
    att = np.vstack([['MODEL'], np.array(coinlist).reshape(-1, 1)])
    
    # Concatenate the coin names and the zeros horizontally and convert to a DataFrame
    we = pd.DataFrame(np.hstack([att, we]))
    
    # For each model in the list of models
    for n, x in enumerate(modellist[:, 1:2]):
        # Initialize a list with the model name and zeros for each coin
        dr = [str(x).strip('[]')] + ['0']*l
        dr = np.array(dr).reshape(-1, 1)
        
        # For each coin in the coinlist
        for nl, y in enumerate(coinlist):
            # Extract the parameters for the model
            d = modellist[n, 2:].tolist()
            
            # Make a prediction for the coin on the given date using the model
            xtest, ct, ytest = Make_Prediction(y, d, date)
            pred = find_finalpo(x, xtest, ct, ytest)
            
            # Store the prediction in the list
            dr[nl + 1, :] = float(pred)
        
        # Store the list of predictions in the DataFrame
        we[n + 1] = dr
    
    # Return the DataFrame of predictions
    return we


# This function takes a model name, a coin name, and two sets of model data as input. 
# It prepares the model data and creates a list that includes the coin name, the model name, and the prepared model data. 
# The function returns this list.
def create_model_list(model, coinname, modeldata1, modeldata2):
    # Define the structure of the list
    structure = ['coinname', 'modelname', 'sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6', 'sc7', 'sc8', 'sc9',
                 'sc10', 'sc11', 'sc12', 'sc13', 'sc14', 'sc15', 'sc16', 'sc17', 'sc18', 'sc19', 'sc20',
                 'sc21', 'sc22', 'sc23', 'sc24', 'sc25', 'sc26', 'sc27', 'sc28', 'sc29', 'sc30', 'sc31',
                 'sc32', 'sc33', 'sc34', 'sc35', 'sc36', 'sc37', 'sc38', 'sc39']

    # Prepare the model data
    _, _, _, _, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, \
    sc9, sc10, sc11, sc12, sc13, sc14, sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, \
    sc23, sc24, sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34, sc35, sc36, \
    sc37, sc38, sc39, _ = prepare(modeldata1, modeldata2)

    # Create the list with the coin name, the model name, and the prepared model data
    structure2 = np.hstack((coinname, model, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8,
                            sc9, sc10, sc11, sc12, sc13, sc14, sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22,
                            sc23, sc24, sc25, sc26, sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34, sc35, sc36,
                            sc37, sc38, sc39))

    # Return the created list
    return structure2


# This function takes multiple models, their corresponding data, a list of coins, and a date as input. 
# It uses the 'create_model_list' function to prepare the data for each model and the 'frommodellist' function to make predictions for each coin on the given date using each model. 
# The function returns the predictions made by all the models for all the coins.
def pool4_12models(model1,model1point1,model1point2,model1point3,model2,model2point1,model2point2,model2point3,model3,model3point1,model3point2,model3point3,modeldata1,model1point1_data,
                  model1point2_data,model1point3_data,modeldata1btc,model1point1btc_data,model1point2btc_data,model1point3btc_data,modeldata2,model2point1_data,model2point2_data,model2point3_data,
                  modeldata2btc,model2point1btc_data,model2point2btc_data,model2point3btc_data,modeldata3,model3point1_data,model3point2_data,model3point3_data,modeldata3btc,model3point1btc_data,
                  model3point2btc_data,model3point3btc_data,coinlist,Date):
    
    # For each model, prepare the data and reshape it into a row vector
    s1 = np.reshape(create_model_list(model1,'ALPHAUSDT', modeldata1, modeldata1btc), (1, -1))
    s1_1 = np.reshape(create_model_list(model1point1, 'ALPHAUSDT', model1point1_data, model1point1btc_data), (1, -1))
    s1_2 = np.reshape(create_model_list(model1point2, 'ALPHAUSDT', model1point2_data, model1point2btc_data), (1, -1))
    s1_3 = np.reshape(create_model_list(model1point3, 'ALPHAUSDT', model1point3_data, model1point3btc_data), (1, -1))
    s2 = np.reshape(create_model_list(model2, 'ALPHAUSDT', modeldata2, modeldata2btc), (1, -1))
    s2_1 = np.reshape(create_model_list(model2point1,'ALPHAUSDT' , model2point1_data, model2point1btc_data), (1, -1))
    s2_2 = np.reshape(create_model_list(model2point2, 'ALPHAUSDT', model2point2_data, model2point2btc_data), (1, -1))
    s2_3 = np.reshape(create_model_list(model2point3, 'ALPHAUSDT', model2point3_data, model2point3btc_data), (1, -1))
    s3 = np.reshape(create_model_list(model3, 'ALPHAUSDT', modeldata3, modeldata3btc), (1, -1))
    s3_1 = np.reshape(create_model_list(model3point1, 'ALPHAUSDT', model3point1_data, model3point1btc_data), (1, -1))
    s3_2 = np.reshape(create_model_list(model3point2, 'ALPHAUSDT', model3point2_data, model3point2btc_data), (1, -1))
    s3_3 = np.reshape(create_model_list(model3point3, 'ALPHAUSDT', model3point3_data, model3point3btc_data), (1, -1))
    
    # Stack all the prepared data vertically
    totalstu = np.vstack((s1,s1_1,s1_2,s1_3,s2,s2_1,s2_2,s2_3,s3,s3_1,s3_2,s3_3))
    
    # Make predictions for each coin on the given date using each model
    opr = frommodellist(totalstu, coinlist, Date)
    
    # Return the predictions
    return opr
result14 = pool4_12models(model1,model1point1,model1point2,model1point3,model2,model2point1,model2point2,model2point3,model3,model3point1,model3point2,model3point3,modeldata1,model1point1_data,
                  model1point2_data,model1point3_data,modeldata1btc,model1point1btc_data,model1point2btc_data,model1point3btc_data,modeldata2,model2point1_data,model2point2_data,model2point3_data,
                  modeldata2btc,model2point1btc_data,model2point2btc_data,model2point3btc_data,modeldata3,model3point1_data,model3point2_data,model3point3_data,modeldata3btc,model3point1btc_data,
                  model3point2btc_data,model3point3btc_data,coinlist,Date)

np.save('lastresult', result14)


from All_common_functions import *
# Define a function that takes a symbol as an argument
def find_position(simbal):
            # Fetch the positions from the futures account
            w = client.futures_account()['positions']
            # Convert the positions to a numpy array
            w = np.array(w)
            # Initialize a variable n to 1
            n = 1
            # Initialize a variable quantity to 0
            quantity = 0
            # Iterate over each position in the array
            for x in w:
                # Extract the symbol from the current position
                ie = x['symbol']
                # Check if the symbol is not equal to the input symbol
                if ie != simbal:
                    # If not, set n to 1
                    n = 1
                # Check if the symbol is equal to the input symbol
                elif ie == simbal:
                    # If so, set n to 0
                    n = 0
                # Check if n is 0
                if n == 0:
                    # If so, set quantity to the notional value of the current position
                    quantity = x['notional']
                    # Break the loop
                    break
            # Convert quantity to a float and take the absolute value
            quantity = abs(float(quantity))
            # Multiply quantity by 30
            quantity = quantity * 30
            # Add 20 to quantity
            quantity = quantity + 20
            # Round quantity to the nearest whole number
            quantity = round(quantity,0)
            # Return quantity
            return quantity
        
# Define a function that takes a symbol, check_switch, normal_position, and position as arguments
def cancel_open_position(symboll,check_switch, normal_position, position):
    # Cancel all open orders for the given symbol
    client.futures_cancel_all_open_orders(symbol=symboll)
    # Check if the position is off
    if position == "off":
        # Check if the check_switch is 'BUY'
        if check_switch == 'BUY':
            # Print a message
            print('Entered before check_switch')
            # Find the position for the given symbol
            we = find_position(symboll)
            # Find the position side for the given symbol
            BE = find_position_side(symboll)
            # Try to place an order
            try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
            # Print a message
            print('OUT OF POSITION')
        # Check if the check_switch is 'SELL'
        elif check_switch == 'SELL':
            # Print a message
            print('Entered before check_switch')
            # Find the position for the given symbol
            we = find_position(symboll)
            # Find the position side for the given symbol
            BE = find_position_side(symboll)
            # Try to place an order
            try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
            # Print a message
            print('OUT OF POSITION ')
    # Set the position to 'on'
    position = 'on'
    # End the function and return
    return
# Define a function that takes a symbol, start time, and risk coin as arguments
def tade_ornontrade(symboll, start_time, riskcoin):
    # Print a message indicating the function has been entered
    print('entered trade or no trade function', flush=True)
    # Get the specific account of the coin for the given symbol and start time
    orr = get_specific_account_of_coin(symboll,start_time)
    # Print the pnl of the given symbol
    print('the pnl of',symboll,'is',orr, flush=True)
    # Initialize a variable endo to 0
    endo = 0
    # Check if the pnl is less than or equal to the risk coin
    if orr <= riskcoin:
        # If so, set endo to 'NONTRADABLE'
        endo = 'NONTRADABLE'
    # Check if the pnl is greater than or equal to the risk coin
    elif orr >= riskcoin:
        # If so, set endo to 'TRADABLE'
        endo = 'TRADABLE'
    # Return endo
    return endo

# Define a function that takes a symbol, check_switch, normal_position, position, and tradable as arguments
def arrest_coin_trading(symboll,check_switch, normal_position, position,tradable):
    # Print a message indicating the function has been entered
    print('entered arrest coin', flush=True)
    # Check if the coin is not tradable and the position is off
    if tradable != 'TRADABLE' and position == "off" :
        # If so, cancel the open position
        cancel_open_position(symboll, check_switch, normal_position, position)
        # Set the position to 'on'
        position = 'on'
        # Print a message indicating the coin has been canceled
        print('entered arrest coin_canceled', flush=True)
    # Check if the coin is tradable
    elif tradable == 'TRADABLE':
        # If so, keep the tradable status as it is
        tradable = tradable
    else:
        # Otherwise, keep the tradable status as it is
        tradable = tradable
    # Return the position
    return position

# Define a function that takes an argument x
def internet_on(x):
    # Try to execute the following block of code
    try:
        # Import data using the input x
        p = import_data(x)
        # Return the imported data
        return p
    # If an exception occurs
    except Exception:
        # Set the internet_check to 'off'
        internet_check = "off"
        # Return the status of the internet_check
        return internet_check

# Define a function that takes several arguments
def taskx(bucket,position,check_switch,symboll,repair,order_switch,tr,ps1,s1,ps2,s2,flb ):
    # Print a message indicating the function has been entered
    print('9', flush=True)
    
    # Check the internet status for the given symbol
    internet_check = internet_on(symboll)
    # Call the onoff function with the internet_check as an argument
    onoff(internet_check)
    # If the internet is on
    if internet_check != "off":
        # Convert the internet_check to a numpy array
        internet_check = np.array(internet_check)
        # Extract the columns from 1 to 5 from the internet_check
        cimpov = internet_check[:, 1:5]
        # Initialize an empty list rg
        rg = []
        # Iterate over the open and close prices in cimpov
        for o, c in zip(cimpov[:, 0:1], cimpov[:, 3:4]):
            # If the open price is greater than the close price
            if float(o) > float(c):
                # Append -1 to rg
                rg.append(-1)
            # If the open price is less than the close price
            if float(o) < float(c):
                # Append 1 to rg
                rg.append(1)
            # If the open price is equal to the close price
            if float(o) == float(c):
                # Append 0 to rg
                rg.append(0)
        # Convert rg to a numpy array
        rg = np.array(rg)
        # Reshape rg to have one column and as many rows as there are elements in rg
        rg = np.reshape(rg, (rg.shape[0], 1))
        # Extract all but the first row from cimpov
        cimpov1 = cimpov[1:, :]
        # Extract all but the last row from rg
        rg1 = rg[:-1, :]
        # Stack rg1 and cimpov1 horizontally
        semi1 = hstack((rg1, cimpov1))
        # Get the number of rows in semi1
        p = semi1.shape[0]
        # Set final_last to flb
        final_last = flb
        # Print the value of final_last
        print('final last is ',flb, flush=True)
        # Convert internet_check to a numpy array
        parr= np.array(internet_check)
        # Print a message indicating the function has reached this point
        print('10', flush=True)
        # Extract the last price from parr
        priceo = float(parr[len(internet_check)-1:len(internet_check),4:5])
        
        # Initialize order_switch to "rr"
        order_switch = "rr"
        # If final_last is 0
        if float(final_last) == 0 :
            # Set order_switch to "SELL"
            order_switch = "SELL"
        # If final_last is 1
        elif float(final_last) == 1:
            # Set order_switch to "BUY"
            order_switch = "BUY"
        # If final_last is -2
        elif float(final_last) == -2:
            # Set order_switch to "TWO"
            order_switch = "TWO"
        # Print a message indicating the function has reached this point
        print('12', flush=True)
        # Pause the execution of the function for 2 seconds
        time.sleep(2)

        # If repair is "ON" and order_switch is equal to check_switch
        if (repair == "ON" and (order_switch == check_switch)):
            # If order_switch is "BUY"
            if order_switch == "BUY":
                # Set check_switch to "SELL"
                check_switch = "SELL"
                # Set repair to "OFF"
                repair = "OFF"
            # If order_switch is "SELL"
            elif order_switch == "SELL":
                # Set check_switch to "BUY"
                check_switch = "BUY"
                # Set repair to "OFF"
                repair = "OFF"
        # Print a message indicating the function has reached this point
        print('13', flush=True)
        # If repair is "OPP" and order_switch is not equal to check_switch
        if (repair == "OPP" and (order_switch != check_switch)):
            # If order_switch is "BUY"
            if  order_switch == "BUY":
                # Set check_switch to "BUY"
                check_switch = "BUY"
                # Set repair to "OFF"
                repair = "OFF"
                # Set position to "off"
                position = "off"
            # If order_switch is "SELL"
            elif order_switch == "SELL":
                # Set check_switch to "SELL"
                check_switch = "SELL"
                # Set repair to "OFF"
                repair = "OFF"
                # Set position to "off"
                position = "off"
        # Set repair to "OFF"
        repair = "OFF"
        # Print a message indicating the function has reached this point
        print('14', flush=True)
        # Print the value of check_switch
        print('check switch is ',check_switch, flush=True)
        # Print the value of order_switch
        print('order_switch is ',order_switch, flush=True)
        # Extract the last open price from cimpov1
        open = float(cimpov1[cimpov1.shape[0] - 1:cimpov1.shape[0], 0:1])
        # Print the value of open
        print('open equals',open, flush=True)
        # If check_switch is "SELL" or "TWO" and order_switch is "BUY"
        if  (check_switch == "SELL" or check_switch == "TWO") and order_switch == "BUY" :
            # Print a message indicating the function has entered this block
            print('enterd inner under orderswitch buy', flush=True)
            # Cancel all orders for the given symbol
            client.futures_cancel_all_open_orders(symbol=symboll)
            # Pause the execution of the function for 1 second
            time.sleep(1)
            # Calculate the quantity to be traded
            cc = quantize(symboll,amount_traded, internet_check)
            # If position is "off"
            if position == "off":
                # Print a message indicating the function has entered this block
                print('Entered before position = off')
                # Find the position for the given symbol
                we = find_position(symboll)
                # Find the position side for the given symbol
                BE = find_position_side(symboll)
                # Print the value of we
                print(we)
                # Try to place an order
                try_order(symboll, BE, we, 911, "TRUE", "TRUE",1)
                # Print a message indicating all positions have been canceled
                print('canceled all positions', flush=True)
            # Set position to "off"
            position = "off"
            # Print a message indicating the function is about to place a buy order
            print('about to make buy order', flush=True)
            # Try to place a buy order
            try_order(symboll, SIDE_BUY, cc, 911, "TRUE", "FALSE",1)
            # Pause the execution of the function for 1 second
            time.sleep(1)
            # Extract the last open price from cimpov1
            open = float(cimpov1[cimpov1.shape[0] - 1:cimpov1.shape[0], 0:1])
            # Calculate s4
            s4 = percentfrominternet(internet_check, 7)
            # Calculate s2
            s2 = open - (s4)
            # Round s2 to 3 decimal places
            s2 = round(s2, 3)
            # Set s1 to s2
            s1 = s2
            # Set tr to 'on'
            tr = 'on'
            # Set check_switch to "BUY"
            check_switch = "BUY"
        # If check_switch is "BUY" or "TWO" and order_switch is "SELL"
        if  (check_switch == "BUY" or check_switch == "TWO") and order_switch == "SELL":
            # Print a message indicating the function has entered this block
            print('enterd inner under orderswitch sell', flush=True)
            # Cancel all orders for the given symbol
            client.futures_cancel_all_open_orders(symbol=symboll)
            # Calculate the quantity to be traded
            aa = quantize(symboll,amount_traded, internet_check)
            # If position is "off"
            if position == "off":
                # Print a message indicating the function has entered this block
                print('Entered before position = off')
                # Find the position for the given symbol
                we = find_position(symboll)
                # Find the position side for the given symbol
                BE = find_position_side(symboll)
                # Print the value of we
                print(we)
                # Try to place an order
                try_order(symboll, BE, we, 911, "TRUE", "TRUE",1)
                # Print a message indicating all positions have been canceled
                print('canceled all positions', flush=True)
            # Set position to "off"
            position = "off"
            # Make a sell order (half usdt)
            print('make sell orders', flush=True)
            # Execute the sell order
            try_order(symboll, SIDE_SELL, aa, 911, "TRUE", "FALSE",1)
            # Pause for 1 second
            time.sleep(1)
            # Get the opening price from the internet check result
            open = float(cimpov1[cimpov1.shape[0] - 1:cimpov1.shape[0], 0:1])
            # Calculate the stop loss price
            s4 = percentfrominternet(internet_check, 7)
            s2 = open + (s4)
            s2 = round(s2, 3)
            s1 = s2
            tr = 'on'
            # Change the check switch status to "SELL"
            check_switch = "SELL"

            # If the order switch is "TWO"
            if order_switch == 'TWO':
                # If the position is "off"
                if  position == 'off':
                    # If the check switch is "BUY"
                    if check_switch == 'BUY':
                        # Cancel all orders
                        client.futures_cancel_all_open_orders(symbol=symboll)
                        print('Entered before check switch')
                        # Find the current position
                        we = find_position(symboll)
                        # Find the current position side
                        BE = find_position_side(symboll)
                        # Execute the order
                        try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
                        print('OUT OF POSITION', flush=True)
                    # If the check switch is "SELL"
                    elif check_switch == 'SELL':
                        # Cancel all orders
                        client.futures_cancel_all_open_orders(symbol=symboll)
                        print('Entered before check switch')
                        # Find the current position
                        we = find_position(symboll)
                        # Find the current position side
                        BE = find_position_side(symboll)
                        # Execute the order
                        try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
                        print('OUT OF POSITION ', flush=True)
                # Set the position to "on"
                position = 'on'
                bucket = 3
                # Change the check switch status to "TWO"
                check_switch ='TWO'

            # If the stop loss price is not 0 and the trade is "on" and the position is 'off'
            if s1 != 0   and tr == "on" and position == 'off' :
                # If the order switch is "BUY"
                if (order_switch == "BUY"):
                    # Calculate the stop loss price
                    dr = max(s1,s2)
                    dr = round(dr, 3)
                    # If the order switch is "BUY" and the price is greater than the stop loss price
                    if (order_switch == "BUY" and priceo > dr):
                        # Cancel all orders
                        client.futures_cancel_all_open_orders(symbol=symboll)
                        # Create a sell order
                        client.futures_create_order(symbol=symboll, side = 'SELL',type='STOP_MARKET', stopPrice= dr,closePosition='true' )
                        print('CANCELED ALL OPEN ORDERS', flush=True)
                    # If the order switch is "BUY" and the price is less than or equal to the stop loss price
                    elif (order_switch == "BUY" and priceo <= dr):
                        print('Entered before order_switch == "BUY"')
                        # Find the current position
                        we = find_position(symboll)
                        # Find the current position side
                        BE = find_position_side(symboll)
                        # Execute the order
                        try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
                        # Set the position to 'on'
                        position = 'on'
                    bucket = 3

                # If the order switch is "SELL"
                if (order_switch == "SELL"):
                    # Calculate the stop loss price
                    dp = min(s1, s2)
                    dp = round(dp, 3)
                    # If the order switch is "SELL" and the price is less than the stop loss price
                    if (order_switch == "SELL" and priceo < dp):
                        # Cancel all orders
                        client.futures_cancel_all_open_orders(symbol=symboll)
                        # Create a buy order
                        client.futures_create_order(symbol=symboll, side = 'BUY',type='STOP_MARKET', stopPrice= dp,closePosition='true' )
                        print('CANCELED ALL OPEN ORDERS', flush=True)
                    # If the order switch is "SELL" and the price is greater than or equal to the stop loss price
                    elif (order_switch == "SELL" and priceo >= dp):
                        print('Entered before order_switch == "BUY"')
                        # Find the current position
                        we = find_position(symboll)
                        # Find the current position side
                        BE = find_position_side(symboll)
                        # Execute the order
                        try_order(symboll, BE, we, 911, "TRUE", "TRUE", 1)
                        # Set the position to 'on'
                        position = 'on'
                    bucket = 3

            pop = 'done'

            # Return the updated values
            return bucket,position,check_switch,symboll,repair,order_switch,tr,ps1,s1,ps2,s2

# This function predicts the trading action (buy, sell, or hold) for a given cryptocurrency using two models and Bayesian inference.
def Total_prediction(COINNAME, COINSCALEDLIST1,COINSCALEDLIST2,model1,model2 ):
    # Wrap the Make_Prediction function
    Make_Prediction =  Wrap_Make_prediction(Make_Prediction)

    # Make predictions for the first coin
    x_test1sushi = Make_Prediction(COINNAME,COINSCALEDLIST1)

    # Make predictions for the second coin
    x_test2sushi = Make_Prediction(COINNAME,COINSCALEDLIST2)

    # Define a function to apply Bayesian inference
    def bayes(re, po):
        bayesset = []
        # For each pair of predictions
        for x, y in zip(re, po):
            x = float(x)
            y = float(y)
            # If both predictions are greater than or equal to 0.5, append 1 to the results
            if x >= 0.5 and y >= 0.5:
                bayesset.append(1)
            # If both predictions are less than 0.5, append 0 to the results
            elif x < 0.5 and y < 0.5:
                bayesset.append(0)
            # Otherwise, append -2 to the results
            else:
                bayesset.append(-2)
        return bayesset

    # Predict the results for the first coin using the first model
    y_test1sushi = model1.predict(x_test1sushi)

    # Predict the results for the second coin using the second model
    y_test2sushi = model2.predict(x_test2sushi)

    # Apply Bayesian inference to the predictions
    finalpred = bayes(y_test1sushi,y_test2sushi)
    realfinal = 0
    # For each result in the final predictions
    for x in finalpred:
        # If the result is 1, set the final result to 1
        if x == 1:
            realfinal = 1
        # If the result is 0, set the final result to 0
        elif x == 0:
            realfinal = 0
        # Otherwise, set the final result to -2
        else:
            realfinal = -2

    # Return the final result
    return realfinal

# This function imports historical data for a given cryptocurrency.
def import_data(coin_name):
    # Get historical klines (candlestick chart data) for the coin from the Binance API
    c_four_hour = client.get_historical_klines(coin_name, Client.KLINE_INTERVAL_30MINUTE, '3208 HOUR ago UTC ')
    # Return the historical data
    return  c_four_hour

# This function finds the precision of the quantity for a given symbol in futures trading.
def find_precision(symbol):
    # Assign the input symbol to a variable
    simbal = symbol
    
    # Get the exchange information from the Binance futures API
    eerr = client.futures_exchange_info()
    
    # Extract the symbols from the exchange information
    ere = eerr['symbols']
    
    # Convert the symbols to a numpy array
    ere = np.array(ere)
    
    # Initialize a flag variable
    n = 1
    
    # Initialize a variable to store the quantity precision
    quantity = 0
    
    # For each symbol in the exchange information
    for x in ere:
        # Extract the symbol
        ie = x['symbol']
        
        # If the symbol does not match the input symbol, set the flag to 1
        if ie != simbal:
            n = 1
        # If the symbol matches the input symbol, set the flag to 0
        elif ie == simbal:
            n = 0
        
        # If the flag is 0, extract the quantity precision and break the loop
        if n == 0:
            quantity = x['quantityPrecision']
            break
    
    # Return the quantity precision
    return quantity


# This function adjusts the quantity of a cryptocurrency to ensure that the total value is above a certain threshold.
def quantize(Symboll, quantity,p):
    # Print a message indicating the start of the function
    print('entered quantize')
    
    # Get the symbol information from the Binance API
    er = client.get_symbol_info(Symboll)
    
    # Extract the filters from the symbol information
    e = er['filters']
    
    # Find the precision of the quantity for the symbol
    PERS = find_precision(Symboll)
    
    # Convert the filters to a numpy array
    ee = np.array(e)
    
    # Extract the first filter from the array
    [eee] = ee[0:1]
    
    # Extract the tick size from the filter and convert it to a float
    wee = eee['tickSize']
    wee = float(wee)
    
    # Convert the input prices to a numpy array
    pe = np.array(p)
    
    # Extract the last price from the array and convert it to a float
    p1 = float(pe[len(p) - 1:len(p), 4:5])
    
    # Calculate the quantity in terms of USDT
    usdt = (quantity/p1)
    
    # Print a message indicating the start of the if statement
    print('about to enter if')
    
    # If the total value in USDT is greater than 6, keep the quantity as is
    if (usdt * p1) > 6:
        usdt = usdt
    # If the total value in USDT is less than 6, adjust the quantity upwards until the total value is at least 6
    elif (usdt * p1) < 6:
        er = 6 / p1
        while usdt < er:
            usdt = usdt + wee
    
    # Round the quantity to the appropriate precision
    usdt = round(usdt,PERS)
    
    # Print a message indicating the end of the function
    print('finished quantize', usdt ,flush=True)
    
    # Return the adjusted quantity
    return usdt

# This function creates a futures order based on the given parameters.
def futures_create_order(symbol,side,quantity,price,market,reduce):
    # If the order is a market order and is a reduce-only order
    if market == "TRUE" and reduce == "TRUE":
        # Create a reduce-only market order
        client.futures_create_order(
            symbol=symbol,
            side=(side),
            quantity=quantity,
            type=FUTURE_ORDER_TYPE_MARKET,
            reduceOnly=True,
        )
    # If the order is a market order and is not a reduce-only order
    if market == "TRUE" and reduce == "FALSE":
        # Create a market order
        client.futures_create_order(
            symbol=symbol,
            side=(side),
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )
    # If the order is a limit order and is not a reduce-only order
    if market == "FALSE" and reduce == "FALSE":
        # Create a limit order
        client.futures_create_order(
            symbol=symbol,
            side=(side),
            quantity=quantity,
            type=FUTURE_ORDER_TYPE_LIMIT,
            price=price,
            timeInForce=TIME_IN_FORCE_GTC,
        )

    # The function does not return anything
    return


# This function disables and then re-enables a network interface if the internet_check parameter is "off".
def onoff(internet_check):
    # Check if the internet_check parameter is "off"
    if internet_check == "off":
        # Disable the "Ethernet 4" network interface
        os.system('netsh interface set interface "Ethernet 4" disabled')
        
        # Wait for 6 seconds
        time.sleep(6)
        
        # Enable the "Ethernet 4" network interface
        os.system('netsh interface set interface "Ethernet 4" enabled')
        
        # Wait for 5 seconds
        time.sleep(5)
        return


# This function calculates a percentage of the last value in the given data.
def percentfrominternet(interneet_chek, percentage):
    # Extract the last value from the second column of the data and convert it to a float
    pr = float(interneet_chek[interneet_chek.shape[0]-1:interneet_chek.shape[0],1:2])
    
    # Calculate the specified percentage of the last value and convert it to a float
    c2 = float((percentage * pr )/100)
    
    # Return the result
    return c2

# This function calculates a percentage of the mark price for a given symbol.
def percentage(symbol,percentage):
    # Get the mark price for the symbol from the Binance futures API
    pr = client.futures_mark_price(symbol=symbol)
    
    # Convert the mark price to a list of items
    pr1 = dict.items(pr)
    
    # Convert the list of items to a list
    pr2 = list(pr1)
    
    # Convert the list to a numpy array
    pr3 = np.array(pr2)
    
    # Extract the second value from the array and convert it to a float
    c0 = float(pr3[1:2, 1:2])
    
    # Calculate the specified percentage of the mark price and convert it to a float
    c2 = ((percentage * c0) / 100)
    
    # Return the result and the mark price
    return c2, c0


# This function gets the current time in Coordinated Universal Time (UTC) and returns the hour and minute.
def createtimedata():
    # Get the current time in UTC
    now = datetime.utcnow()
    
    # Extract the hour from the current time and convert it to an integer
    nowhour = int(now.strftime("%H"))
    
    # Extract the minute from the current time and convert it to an integer
    nowminute = int(now.strftime("%M"))
    
    # Return the hour and minute
    return nowhour, nowminute

# This function calculates the average movement of a given symbol based on historical data.
def avrage_movment(symbol):
    # Get the historical data for the symbol from the Binance API
    hour = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, '28 HOUR ago UTC ')
    
    # Convert the historical data to a numpy array
    hour = np.array(hour)
    
    # Exclude the last row from the data
    hour = hour[:-1,:]
    
    # Initialize a variable to store the total movement
    total = 0
    
    # Initialize an empty list to store the percentage movements
    p = []
    
    # Extract the opening prices from the data and convert them to floats
    eu = np.array((hour[:, 1:2]).astype(float))
    
    # Extract the closing prices from the data and convert them to floats
    ru = (hour[:,4:5]).astype(float)
    
    # For each pair of opening and closing prices
    for x,y in zip(eu,ru):
        # Calculate the percentage movement, append it to the list
        p.append (float(abs((y - x)/y) * 100))
    
    # For each percentage movement in the list
    for x in p:
        # Add the percentage movement to the total
        total = x + total
    
    # Calculate the average movement by dividing the total by the number of intervals
    r = total/6
    
    # Return the average movement
    return r

# This function calculates the closest value to 30 or 60 for a given input value.
def closest_value(input_value):
    # Initialize a variable to store the result
    tim = 0
    
    # If the input value is less than or equal to 30
    if input_value <= 30:
        # Calculate the difference between 30 and the input value
        tim = 30 - input_value
    # If the input value is greater than 30
    elif input_value > 30:
        # Calculate the difference between 60 and the input value
        tim = 60 - input_value
    
    # Return the result
    return tim

# This function continuously calculates the average movement of a given symbol until it is greater than or equal to 1.1.
def loopavg(symbol):
    # Calculate the average movement of the symbol
    avg_movement = avrage_movment(symbol)
    
    # If the average movement is greater than or equal to 1.1
    if avg_movement >= 1.1:
        # Print the average movement
        print(avg_movement, flush=True)
    # If the average movement is less than 1.1
    else:
        # Call the function again
        loopavg(symbol)

# This function attempts to create a futures order and retries in case of an exception.
def try_order(symbol,side,quantity,price,market,reduce,one):
    # Initialize a variable with the input value
    timee = one
    
    # Try to create a futures order
    try:
        futures_create_order(symbol,side,quantity,price,market,reduce)
        return
    # If an exception occurs
    except Exception:
        # Wait for 1 second
        time.sleep(1)
        
        # Increment the variable by 2
        timee = timee + 2
        
        # Retry the function with the incremented variable
        try_order(symbol,side,quantity,price,market,reduce,timee)
        return

# This function finds the position size for a given symbol.
def find_position_size(simbal):
    # Get the positions from the Binance futures account
    w = client.futures_account()['positions']
    
    # Convert the positions to a numpy array
    w = np.array(w)
    
    # Initialize a flag variable
    n = 1
    
    # Initialize a variable to store the quantity
    quantity = 0
    
    # For each position in the account
    for x in w:
        # Extract the symbol from the position
        ie = x['symbol']
        
        # If the symbol does not match the input symbol, set the flag to 1
        if ie != simbal:
            n = 1
        # If the symbol matches the input symbol, set the flag to 0
        elif ie == simbal:
            n = 0
        
        # If the flag is 0, extract the quantity from the position and break the loop
        if n == 0:
            quantity = x
            break
    
    # Convert the quantity to a pandas DataFrame and then to a numpy array
    quantity = pd.DataFrame(quantity, index=[0])
    quantity = np.array(quantity)
    
    # Extract the price from the quantity and convert it to a float
    [[price]] = quantity[:, 12:13]
    price = float(price)
    
    # Return the price
    return price

# This function checks and repairs the trading position based on various conditions.
def check_repair(symboll, s1, s2 , repair, check_switch,position,COINSCALEDLIST1,COINSCALEDLIST2,model1,model2 ):
    
    
    # Initialize a variable with a default value
    flb = 99
    
    # Check the internet connection
    internet_check = internet_on(symboll)
    
    # If the internet connection is off, turn it on
    onoff(internet_check)
    
    # If the internet connection is off
    if internet_check == 'off':
        # Call the function again
        check_repair(symboll, s1, s2, repair, check_switch, position,COINSCALEDLIST1,COINSCALEDLIST2,model1,model2 )
    
    # If the internet connection is on
    if internet_checkk != "off":
        # Convert the internet check result to a numpy array
        R = np.array(internet_check)
        
        # Extract the high and low values from the result and round them to 3 decimal places
        high = float(R[R.shape[0] - 2:R.shape[0] - 1, 2:3])
        low = float(R[R.shape[0] - 2:R.shape[0] - 1, 3:4])
        high = round(high, 3)
        low = round(low, 3)
        
        # Initialize a flag variable
        fl = 0
        
        # Set the flag based on the check switch
        if check_switch == 'BUY':
            fl = 1
        elif check_switch == 'SELL':
            fl = 0
        elif check_switch =='TWO':
            fl = -2
        
        # Calculate the total prediction
        flb = Total_prediction(symboll,COINSCALEDLIST1,COINSCALEDLIST2,model1,model2)
        
        # Round the input values to 3 decimal places
        s1 = round(s1, 3)
        s2 = round(s2,3)
        
        # If the check switch is 'BUY'
        if check_switch == 'BUY':
            # If the low value is less than or equal to the maximum of s1 and s2
            if low <= float(max(s1, s2)):
                # Set the repair and position to 'ON'
                repair = 'ON'
                position = 'on'
            # If the low value is greater than the maximum of s1 and s2 and the flag does not match the total prediction
            elif low > float(max(s1, s2)) and fl != flb:
                # Set the repair to 'ON' and the position to 'off'
                repair = 'ON'
                position = "off"
            # If the low value is greater than the maximum of s1 and s2 and the flag matches the total prediction
            elif low > float(max(s1, s2)) and fl == flb:
                # Set the repair and position to 'OFF'
                repair = 'OFF'
                position = "off"
        
        # If the check switch is 'SELL'
        if check_switch == 'SELL':
            # If the high value is greater than or equal to the minimum of s1 and s2
            if high >= float(min(s1, s2)):
                # Set the repair and position to 'ON'
                repair = 'ON'
                position = 'on'
            # If the high value is less than the minimum of s1 and s2 and the flag does not match the total prediction
            elif high < float(min(s1, s2)) and fl != flb:
                # Set the repair to 'ON' and the position to 'off'
                repair = 'ON'
                position = "off"
            # If the high value is less than the minimum of s1 and s2 and the flag matches the total prediction
            elif high < float(min(s1, s2)) and fl == flb:
                # Set the repair and position to 'OFF'
                repair = 'OFF'
                position = "off"
        
        # If the check switch is 'TWO'
        if check_switch == 'TWO':
            # If the flag does not match the total prediction
            if fl != flb:
                # Set the repair to 'ON'
                repair = 'ON'
                # Keep the position as it is
                position = position
            # If the flag matches the total prediction
            elif fl == flb:
                # Set the repair to 'OFF'
                repair = 'OFF'
                # Keep the position as it is
                position = position
        
        # Find the position size for the symbol
        price = find_position_size(symboll)
        
        # If the price is greater than 0.1
        if price > 0.1:
            # Set the position to 'off'
            position = "off"
        # If the price is less than or equal to 0
        elif price <= 0:
            # Set the position to 'on'
            position = 'on'
    
    # If the internet connection is off
    elif internet_checkk == "off":
        # Call the function again
        check_repair(symboll, s1, s2, repair, check_switch,position,COINSCALEDLIST1,COINSCALEDLIST2,model1,model2)
    
    # Return the repair, position, and total prediction
    return repair, position, flb

# This function finds the entry price and profit/loss for a given symbol.
def find_entry_price(simbal):
    # Get the positions from the Binance futures account
    w = client.futures_account()['positions']
    
    # Convert the positions to a numpy array
    w = np.array(w)
    
    # Initialize a flag variable
    n = 1
    
    # Initialize a variable to store the quantity
    quantity = 0
    
    # For each position in the account
    for x in w:
        # Extract the symbol from the position
        ie = x['symbol']
        
        # If the symbol does not match the input symbol, set the flag to 1
        if ie != simbal:
            n = 1
        # If the symbol matches the input symbol, set the flag to 0
        elif ie == simbal:
            n = 0
        
        # If the flag is 0, extract the quantity from the position and break the loop
        if n == 0:
            quantity = x
            break
    
    # Convert the quantity to a pandas DataFrame and then to a numpy array
    quantity = pd.DataFrame(quantity, index=[0])
    quantity = np.array(quantity)
    
    # Extract the price from the quantity and convert it to a float
    [[price]] = quantity[:, 8:9]
    price = float(price)
    
    # Extract the profit/loss from the quantity and convert it to a float
    [[pnl]] = quantity[:, 3:4]
    pnl = float(pnl)
    
    # Return the price and profit/loss
    return price, pnl

# This function determines the side of a position for a given symbol based on various conditions.
def find_position_side(simbal):
    # Get the entry price and profit/loss for the symbol
    entryprice, pnl = find_entry_price(simbal)
    
    # Get the ticker symbol from the Binance API
    SYMB = client.get_symbol_ticker()
    
    # Convert the ticker symbol to a pandas DataFrame
    SYMB1  = pd.DataFrame(SYMB)
    
    # Set the index of the DataFrame to the symbol
    SYMB1 = SYMB1.set_index('symbol')
    
    # Get the price for the 'SUSHIUSDT' symbol
    p = SYMB1['price']['SUSHIUSDT']
    
    # Initialize a variable to store the side
    side = 0
    
    # If the profit/loss is greater than or equal to 0 and the price is greater than or equal to the entry price
    if float(pnl) >=0 and float(p) >= float(entryprice):
        # Print 1
        print(1)
        
        # Set the side to sell
        side = SIDE_SELL
    
    # If the profit/loss is greater than or equal to 0 and the price is less than the entry price
    elif  float(pnl) >=0 and float(p) < float(entryprice):
        # Print 2
        print(2)
        
        # Set the side to buy
        side = SIDE_BUY
    
    # If the profit/loss is less than 0 and the price is greater than or equal to the entry price
    elif float(pnl) < 0 and float(p) >= float(entryprice):
        # Print 3
        print(3)
        
        # Set the side to buy
        side = SIDE_BUY
    
    # If the profit/loss is less than 0 and the price is less than the entry price
    elif  float(pnl) < 0 and float(p) < float(entryprice):
        # Print 4
        print(4)
        
        # Set the side to sell
        side = SIDE_SELL
    
    # Return the side
    return side

# This function calculates the profit from a given array of trading data.
def calculate_profit(array):
    # Print a message indicating that the function has been entered
    print('entered_calculate_profit', flush=True)
    
    # Convert the input array to a numpy array
    array = np.array(array)
    
    # Initialize an empty list to store the profit and loss percentages
    percentage = []
    
    # For each pair of corresponding elements in the 7th and 9th columns of the array
    for x, y in zip(array[:,6:7],array[:,8:9]):
        # Calculate the profit and loss percentage and append it to the list
        percentage.append(100 * (float(x)/float(y)))
    
    # Initialize an empty list to store the commission percentages
    percentagecom = []
    
    # For each pair of corresponding elements in the 10th and 9th columns of the array
    for x, y in zip(array[:, 9:10], array[:, 8:9]):
        # Calculate the commission percentage and append it to the list
        percentagecom.append(100 * (float(x) / float(y)))
    
    # Initialize a variable to store the sum of the profit and loss percentages
    sumpercent = 0
    
    # For each profit and loss percentage in the list
    for x in percentage:
        # Add the profit and loss percentage to the sum
        sumpercent = sumpercent + float(x)
    
    # Initialize a variable to store the sum of the commission percentages
    sumcommision = 0
    
    # For each commission percentage in the list
    for x in percentagecom:
        # Add the commission percentage to the sum
        sumcommision = sumcommision + float(x)
    
    # Calculate the final profit by subtracting the sum of the commission percentages from the sum of the profit and loss percentages
    final = sumpercent - sumcommision
    
    # Return the final profit
    return final

# This function retrieves the profit and loss (PNL) for a given symbol within a specified time range.
def get_pnl(simbol,start_time, end_time):
    # Get the futures account trades for the symbol from the Binance API within the specified time range
    drip = client.futures_account_trades(symbol= simbol, start_str=start_time, endTime=end_time)
    
    # Convert the trades to a pandas DataFrame
    drip2 = pd.DataFrame(drip)
    
    # Return the DataFrame
    return drip2

# This function retrieves the specific account of a coin and calculates the profit.
def get_specific_account_of_coin(simbol,start_time):
    # Get the current time in UTC
    now = datetime.utcnow()
    
    # Convert the current time to a timestamp
    now12 = datetime.timestamp(now)
    
    # Multiply the timestamp by 1000 and round it to the nearest whole number
    now2 = now12 *1000
    now2 = int(round(now2,0))
    
    # Get the profit and loss for the symbol within the specified time range
    w = get_pnl(simbol, start_time, now2)
    
    # Convert the profit and loss to a numpy array
    w = np.array(w)
    
    # Extract the first element from the 12th column of the array
    [[first]] = w[0:1, 11:12]
    
    # Initialize a variable to store the start index
    start = 0
    
    # Initialize a copy of the array
    nw = w
    
    # Initialize a variable with the first element
    first1 = first
    
    # Initialize a variable with the first element
    now3 = first
    
    # If the start time is less than the first element
    if start_time< first:
        # While the start time is less than the first element
        while  start_time< first1:
            # Get the profit and loss for the symbol within the specified time range
            w1 = get_pnl(simbol, start_time, now3)
            
            # Convert the profit and loss to a numpy array
            w1 = np.array(w1)
            
            # Extract the first element from the 12th column of the array
            [[first1]] = w1[0:1, 11:12]
            
            # If the first element has changed
            if now3 != first1:
                # Update the first element
                now3 = first1
                
                # Stack the new profit and loss on top of the existing array
                nw = vstack((w1, nw))
            # If the first element has not changed, break the loop
            else: break
    
    # Extract the first element from the 12th column of the array
    [[first]] = nw[0:1, 11:12]
    
    # If the start time is greater than or equal to the first element
    if start_time >= first:
        # Initialize a counter variable
        n = 0
        
        # Initialize a flag variable
        close = 1
        
        # For each element in the 12th column of the array
        for x in nw[:, 11:12]:
            # Increment the counter
            n = n + 1
            
            # Extract the element
            [e] = x
            
            # If the element is greater than or equal to the start time and the flag is 1
            if e >= start_time and close == 1:
                # Update the start index with the counter
                start = n
                
                # Set the flag to 0
                close = 0
            # If the element is less than the start time, keep the start index as it is
            else: start = start
        
        # Extract the subarray from the start index to the end of the array
        new_array = nw[start:,:]
        
        # Calculate the profit from the subarray
        nWw = calculate_profit(new_array)
    
    # Return the profit
    return nWw

def get_account_balance_percentage(initial_number):
    # Fetch the account balance from the client
    pr = client.futures_account_balance()
    
    # Convert the account balance data into a pandas DataFrame
    prmmpi = pd.DataFrame(pr)
    
    # Set the index of the DataFrame to the 'asset' column
    prmmpi = prmmpi.set_index(prmmpi.loc[:,"asset"])
    
    # Get the balance of USDT (Tether, a type of cryptocurrency)
    usdt = prmmpi.loc["USDT"]["balance"]
    
    # Convert the USDT balance to a float
    usdt = float(usdt)
    
    # Calculate the difference between the initial number and the USDT balance
    op = initial_number - usdt
    
    # Calculate the absolute percentage difference
    ex = (100 * abs(op)) / initial_number
    
    # Initialize nf_ex to 0
    nf_ex = 0
    
    # If the difference is positive or zero, nf_ex is the negative of the percentage difference
    if op >= 0:
        nf_ex = -ex
    # If the difference is negative, nf_ex is the absolute value of the percentage difference
    elif op < 0:
        nf_ex = abs(ex)
    
    # Return the final calculated value
    return nf_ex


pr = client.futures_account_balance()

dateoftraining = ["26 NOV, 2020", "26 NOV , 2022"]
COINNAMESUSHI = 'SUSHIUSDT'
COINDATASUSHI1 =  dateoftraining[0]
COINDATASUSHI2 = dateoftraining[1]
COINNAMETHETA = 'ETCUSDT'
COINDATATHETA1 = dateoftraining[0]
COINDATATHETA2 = dateoftraining[1]
COINNAMEALPHA = 'ALPHAUSDT'
COINDATAALPHA1 = dateoftraining[0]
COINDATAALPHA2 = dateoftraining[1]
COINNAMEALGO = 'ALGOUSDT'
COINDATAALGO1 = dateoftraining[0]
COINDATAALGO2 = dateoftraining[1]
COINNAMEWAVES = 'WAVESUSDT'
COINDATAWAVES1 = dateoftraining[0]
COINDATAWAVES2 = dateoftraining[1]
d = np.load('Data_array.npy')
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
model1scales = np.load('model1scales.npy')
model2scales = np.load('model2scales.npy')
model3scales = np.load('model3scales.npy')
days_plot4 = np.load('lastresult.npy')
[[de]]= days_plot4[0:1,1:2]
[[model1]] = days_plot4[0:1,1:2] #'1-'
[[model1point1]] = days_plot4[0:1,2:3] #'1-'
[[model1point2]] = days_plot4[0:1,3:4]  #'1-'
[[model1point3]] = days_plot4[0:1,4:5] #'0'
[[model2]] = days_plot4[0:1,5:6] #'1-'
[[model2point1]] = days_plot4[0:1,6:7]  #'1-'
[[model2point2]] = days_plot4[0:1,7:8] #'1-'
[[model2point3]] = days_plot4[0:1,8:9]  #'0'
[[model3]] = days_plot4[0:1,9:10] #'1-'
[[model3point1]] = days_plot4[0:1,10:11]  #'1-'
[[model3point2]] = days_plot4[0:1,11:12]  #'1-'
[[model3point3]] = days_plot4[0:1,12:13]  #'0'
#trained on SUSHI SEP 20 2020 TO MARCH 26 2022 37 TRAITS
model3point3 = load_model(model3point3)
model1point3 = load_model(model1point3)
model2point3 = load_model(model2point3)
model1point1 = load_model(model1point1)
model1 = load_model(model1)
Data_array = np.load('Data_array.npy')
# Changable variables
bucket1 = int(Data_array[0:1,1:2])
bucket2 = int(Data_array[0:1,2:3])
bucket3 = int(Data_array[0:1,3:4])
bucket4 = int(Data_array[0:1,4:5])
bucket5 = int(Data_array[0:1,5:6])
bucket6 = int(Data_array[0:1,6:7])
bucket7 = int(Data_array[0:1,7:8])
bucket8 = int(Data_array[0:1,8:9])
[[position1]] = Data_array[1:2,1:2]
position1 = str(position1)
[[position2]] = Data_array[1:2,2:3]
position2 = str(position2)
[[position3]] = Data_array[1:2,3:4]
position3 = str(position3)
[[position4]] = Data_array[1:2,4:5]
position4 = str(position4)
[[position5]] = Data_array[1:2,5:6]
position5 = str(position5)
[[position6]] = Data_array[1:2,6:7]
position6 = str(position6)
[[position7]] = Data_array[1:2,7:8]
position7 = str(position7)
[[position8]] = Data_array[1:2,8:9]
position8 = str(position8)
[[check_switch1]] = Data_array[2:3,1:2]
check_switch1 = str(check_switch1)
[[check_switch2]] = Data_array[2:3,2:3]
check_switch2 = str(check_switch2)
[[check_switch3]] = Data_array[2:3,3:4]
check_switch3 = str(check_switch3)
[[check_switch4]] = Data_array[2:3,4:5]
check_switch4 = str(check_switch4)
[[check_switch5]] = Data_array[2:3,5:6]
check_switch5 = str(check_switch5)
[[check_switch6]] = Data_array[2:3,6:7]
check_switch6 = str(check_switch6)
[[check_switch7]] = Data_array[2:3,7:8]
check_switch7 = str(check_switch7)
[[check_switch8]] = Data_array[2:3,8:9]
check_switch8 = str(check_switch8)
symboll1 = COINNAMETHETA
symboll2 = COINNAMEALPHA
symboll3 = COINNAMESUSHI
symboll4 = COINNAMEALGO
symboll5 = COINNAMEWAVES
symboll6 = "WAVESUSDT"
symboll7 = "GALAUSDT"
symboll8 = "LTCUSDT"
FIRST_COINSCALED1 = model1scales
SECOND_COINSCALED1 = model2scales
THIRD_COINSCALED1 = model1scales
FOURTH_COINSCALED1 = 0
FIFTH_COINSCALED1 = 0
SIXTH_COINSCALED1 = 0
SEVENTH_COINSCALED1 = 0
EIGHTH_COINSCALED1= 0
FIRST_COINSCALED2 = model2scales
SECOND_COINSCALED2 = model1scales
THIRD_COINSCALED2 = model3scales
FOURTH_COINSCALED2 = 0
FIFTH_COINSCALED2 = 0
SIXTH_COINSCALED2 = 0
SEVENTH_COINSCALED2 = 0
EIGHTH_COINSCALED2= 0
FIRST_MODEL1 = model1point3
SECOND_MODEL1 = model2point3
THIRD_MODEL1 = model1
FOURTH_MODEL1 = 0
FIFTH_MODEL1 = 0
SIXTH_MODEL1 = 0
SEVENTH_MODEL1 = 0
EIGHTH_MODEL1= 0
FIRST_MODEL2= model2point3
SECOND_MODEL2= model1point1
THIRD_MODEL2= model3point3
FOURTH_MODEL2= 0
FIFTH_MODEL2= 0
SIXTH_MODEL2= 0
SEVENTH_MODEL2= 0
EIGHTH_MODEL2= 0
[[repair1]] = Data_array[3:4,1:2]
repair1 = str(repair1)
[[repair2]] = Data_array[3:4,2:3]
repair2 = str(repair2)
[[repair3]] = Data_array[3:4,3:4]
repair3 = str(repair3)
[[repair4]] = Data_array[3:4,4:5]
repair4 = str(repair4)
[[repair5]] = Data_array[3:4,5:6]
repair5 = str(repair5)
[[repair6]] = Data_array[3:4,6:7]
repair6 = str(repair6)
[[repair7]] = Data_array[3:4,7:8]
repair7 = str(repair7)
[[repair8]] = Data_array[3:4,8:9]
repair8 = str(repair8)
order_switch1 = "rr"
order_switch2 = "rr"
order_switch3 = "rr"
order_switch4 = "rr"
order_switch5 = "rr"
order_switch6 = "rr"
order_switch7 = "rr"
order_switch8 = "rr"
[[tr1]] = Data_array[4:5,1:2]
tr1 = str(tr1)
[[tr2]] = Data_array[4:5,2:3]
tr2 = str(tr2)
[[tr3]] = Data_array[4:5,3:4]
tr3 = str(tr3)
[[tr4]] = Data_array[4:5,4:5]
tr4 = str(tr4)
[[tr5]] = Data_array[4:5,5:6]
tr5 = str(tr5)
[[tr6]] = Data_array[4:5,6:7]
tr6 = str(tr6)
[[tr7]] = Data_array[4:5,7:8]
tr7 = str(tr7)
[[tr8]] = Data_array[4:5,8:9]
tr8 = str(tr8)
p1s1 =int(Data_array[5:6,1:2])
p2s1 = int(Data_array[5:6,2:3])
p3s1 = int(Data_array[5:6,3:4])
p4s1 = int(Data_array[5:6,4:5])
p5s1 = int(Data_array[5:6,5:6])
p6s1 = int(Data_array[5:6,6:7])
p7s1 = int(Data_array[5:6,7:8])
p8s1 = int(Data_array[5:6,8:9])
flb1 = int(Data_array[6:7,1:2])
flb2 = int(Data_array[6:7,2:3])
flb3 = int(Data_array[6:7,3:4])
flb4 = int(Data_array[6:7,4:5])
flb5 = int(Data_array[6:7,5:6])
flb6 = int(Data_array[6:7,6:7])
flb7 = int(Data_array[6:7,7:8])
flb8 = int(Data_array[6:7,8:9])
s11 = float(Data_array[7:8,1:2])
s12 = float(Data_array[7:8,2:3])
s13 = float(Data_array[7:8,3:4])
s14 = float(Data_array[7:8,4:5])
s15 = float(Data_array[7:8,5:6])
s16 = float(Data_array[7:8,6:7])
s17 = float(Data_array[7:8,7:8])
s18 = float(Data_array[7:8,8:9])
p1s2 = float(Data_array[8:9,1:2])
p2s2 = float(Data_array[8:9,2:3])
p3s2 = float(Data_array[8:9,3:4])
p4s2 = float(Data_array[8:9,4:5])
p5s2 = float(Data_array[8:9,5:6])
p6s2 = float(Data_array[8:9,6:7])
p7s2 = float(Data_array[8:9,7:8])
p8s2 = float(Data_array[8:9,8:9])
s21 = float(Data_array[9:10,1:2])
s22 = float(Data_array[9:10,2:3])
s23 = float(Data_array[9:10,3:4])
s24 = float(Data_array[9:10,4:5])
s25 = float(Data_array[9:10,5:6])
s26 = float(Data_array[9:10,6:7])
s27 = float(Data_array[9:10,7:8])
s28 = float(Data_array[9:10,8:9])
tradable1 = 'TRADABLE'
tradable2 = 'TRADABLE'
tradable3 = 'TRADABLE'
tradable4 = 'TRADABLE'
tradable5 = 'TRADABLE'
tradable6 = 'TRADABLE'
tradable7 = 'TRADABLE'
tradable8 = 'TRADABLE'
# START OF MONTH NUMBERS, (DATE OF THE START OF THE MONTH AND BALANCE OF THE ACCOUNT ON THE START OF THE MONTH )
date_start_of_month = 1674969325000
balance_at_the_start_of_month = 59
riskp = -20
riskcoin = -30
amount_traded = 7
while True:
    print('1', flush=True)
    def internet_on(x):
        try:
            p = import_data(x)
            return p
        except Exception:
            internet_check = "off"
            return internet_check
    print('2', flush=True)
    internet_checkk = internet_on('BTCUSDT')
    onoff(internet_checkk)
    internet_checkk = internet_on('BTCUSDT')
    print('3', flush=True)
    if internet_checkk != "off":
        hour, minute = createtimedata()
        print('Check risk parameters for whole account', flush=True)
        ors = get_account_balance_percentage(balance_at_the_start_of_month )
        print(ors, flush=True)
        print(riskp, flush=True)
        if ors <= riskp:

            cancel_open_position(symboll1,check_switch1,7,position1)
            cancel_open_position(symboll2, check_switch2, 7, position2)
            cancel_open_position(symboll3, check_switch3, 7, position3)
            break
        tradable1 = tade_ornontrade(symboll1,date_start_of_month,riskcoin)
        print('THETA TRADABLE IS ',tradable1, flush=True)
        if tradable1 != 'TRADABLE':
            bucket1 = 3
        else: bucket1 = bucket1
        tradable2 = tade_ornontrade(symboll2, date_start_of_month, riskcoin)
        print('ALPHA TRADABLE IS ',tradable2, flush=True)
        if tradable2 != 'TRADABLE':
            bucket2 = 3
            print('after alpha trade deal', flush=True)
        else: bucket2 = bucket2
        tradable3 = tade_ornontrade(symboll3, date_start_of_month, riskcoin)
        print('SUSHI TRADABLE IS ',tradable3, flush=True)
        if tradable3 != 'TRADABLE':
            bucket3 = 3
        else: bucket3 = bucket3
        print('about to pass arresting', flush= True)
        position1 =arrest_coin_trading(symboll1, check_switch1, 7, position1,tradable1)
        position2 = arrest_coin_trading(symboll2, check_switch2, 7, position2, tradable2)
        position3 = arrest_coin_trading(symboll3, check_switch3, 7, position3, tradable3)
        if bucket2  and  bucket3 and bucket1  >= 3:
            print('after all buckets are 3', flush=True)
            time.sleep(60)
            hour, minute = createtimedata()
            near = closest_value(minute)
            r2 = (near + 1)
            r3 = (r2) * 60
            print('ready to sleep for', r3, 'seconds', flush=True)
            time.sleep(r3)
            print('awoken up', flush=True)

            internet_check = internet_on('BTCUSDT')
            onoff(internet_check)
            internet_check = internet_on('BTCUSDT')
            if internet_check != "off":
                print('about to fix repair', flush=True)
                if tradable1 =='TRADABLE':
                    repair1,position1,flb1 =check_repair(symboll1,s11,s21, repair1,check_switch1,position1,FIRST_COINSCALED1,FIRST_COINSCALED2,FIRST_MODEL1,FIRST_MODEL2)
                    # repair1,position1,flb1
                    Data_array[3:4,1:2] = repair1
                    Data_array[1:2,1:2]  = position1
                    Data_array[6:7, 1:2] = flb1
                os.remove('Data_array.npy')
                np.save('Data_array',Data_array)
                print('THETA repair is', repair1, flush=True)
                print('THETA position is', position1, flush=True)
                if tradable2 =='TRADABLE':
                    repair2,position2,flb2 = check_repair(symboll2, s12, s22, repair2, check_switch2,position2,SECOND_COINSCALED1,SECOND_COINSCALED2,SECOND_MODEL1,SECOND_MODEL2)
                    # update repair2,position2,flb2
                    Data_array[3:4, 2:3] = repair2
                    Data_array[1:2, 2:3] = position2
                    Data_array[6:7, 2:3] = flb2
                os.remove('Data_array.npy')
                np.save('Data_array', Data_array)
                print('ALPHA repair is', repair2, flush=True)
                print('ALPHA position is', position2, flush=True)
                if tradable3 =='TRADABLE':
                    repair3,position3,flb3 = check_repair(symboll3, s13,s23, repair3, check_switch3,position3,THIRD_COINSCALED1,THIRD_COINSCALED2,THIRD_MODEL1,THIRD_MODEL2)
                    # update repair3,position3,flb3
                    Data_array[3:4, 3:4] = repair2
                    Data_array[1:2, 3:4] = position2
                    Data_array[6:7, 3:4] = flb2
                os.remove('Data_array.npy')
                np.save('Data_array', Data_array)
                print('SUSHI repair is', repair3, flush=True)
                print('SUSHI position is', position3, flush=True)
                print('fixed repair')
            bucket1 = 0
            bucket2 = 0
            bucket3 = 0
            bucket4 = 0
            bucket5 = 0
            bucket6 = 0
            bucket7 = 0
            bucket8 = 3
            buck_array = [0,0,0,0,0,0,0,3]
            buck_array = np.array(buck_array)
            Data_array[0:1,1:] = buck_array
            tr1 = "off"
            tr2 = "off"
            tr3 = "off"
            tr4 = "off"
            tr5 = "off"
            tr6 = "off"
            tr7 = "off"
            tr8 = "off"
            tr_array=["off","off","off","off","off","off","off","off"]
            tr_array = np.array(tr_array)
            Data_array[4:5,1:] = tr_array
            order_switch1 = "rr"
            order_switch2 = "rr"
            order_switch3 = "rr"
            order_switch4 = "rr"
            order_switch5 = "rr"
            order_switch6 = "rr"
            order_switch7 = "rr"
            order_switch8 = "rr"
            p1s1 = 0
            p2s1 = 0
            p3s1 = 0
            p4s1 = 0
            p5s1 = 0
            p6s1 = 0
            p7s1 = 0
            p8s1 = 0
            ps1_array = [0, 0, 0, 0, 0, 0, 0, 0]
            ps1_array = np.array(ps1_array)
            Data_array[5:6, 1:] = ps1_array
            # update the previous variables
            os.remove('Data_array.npy')
            np.save('Data_array', Data_array)
            

        if bucket1 < 3 and tradable1 == 'TRADABLE':
            print('6', flush=True)
            if flb1 == 99:
                flb1 = Total_prediction(symboll1,FIRST_COINSCALED1,FIRST_COINSCALED2,FIRST_MODEL1,FIRST_MODEL2)
                # update flb1
            else:
                flb1 = flb1
            if repair1 == 'OFF':
                bucket1 = 3
            else:
                print('7', flush=True)
                bucket1,position1,check_switch1,symboll1,repair1,order_switch1,tr1,p1s1,s11,p1s2,s21 = taskx(bucket1,position1,check_switch1,symboll1,repair1,order_switch1,tr1,p1s1,s11,p1s2,s21,flb1)
                # update  bucket1,position1,check_switch1,symboll1,repair1,order_switch1,tr1,p1s1,s11,p1s2,s21
                Data_array[0:1, 1:2] = bucket1
                Data_array[1:2, 1:2] = position1
                Data_array[2:3, 1:2] = check_switch1
                Data_array[3:4, 1:2] = repair1
                Data_array[4:5, 1:2] = tr1
                Data_array[5:6, 1:2] = p1s1
                Data_array[7:8, 1:2] = s11
                Data_array[8:9, 1:2] = p1s2
                Data_array[9:10, 1:2] = s21
                os.remove('Data_array.npy')
                np.save('Data_array', Data_array)
        if bucket2 < 3 and tradable2 == 'TRADABLE' :
            print('6.2', flush=True)
            if flb2 == 99:
                flb2 = Total_prediction(symboll2,SECOND_COINSCALED1,SECOND_COINSCALED2,SECOND_MODEL1,SECOND_MODEL2)
                # update flb2
            else:
                flb2 = flb2
            if repair2 == 'OFF':
                bucket2 = 3
            else:
                print('7.2', flush=True)
                bucket2,position2,check_switch2,symboll2,repair2,order_switch2,tr2,p2s1,s12,p2s2,s22 = taskx(bucket2,position2,check_switch2,symboll2,repair2,order_switch2,tr2,p2s1,s12,p2s2,s22,flb2)
                # update  bucket2,position2,check_switch2,symboll2,repair2,order_switch2,tr2,p2s1,s12,p2s2,s22
                Data_array[0:1, 2:3] = bucket1
                Data_array[1:2, 2:3] = position1
                Data_array[2:3, 2:3] = check_switch1
                Data_array[3:4, 2:3] = repair1
                Data_array[4:5, 2:3] = tr1
                Data_array[5:6, 2:3] = p1s1
                Data_array[7:8, 2:3] = s11
                Data_array[8:9, 2:3] = p1s2
                Data_array[9:10, 2:3] = s21
                os.remove('Data_array.npy')
                np.save('Data_array', Data_array)
        if bucket3 < 3 and tradable3 == 'TRADABLE':
            print('6.3', flush=True)
            if flb3 == 99:
                flb3 = Total_prediction(symboll3,THIRD_COINSCALED1,THIRD_COINSCALED2,THIRD_MODEL1,THIRD_MODEL2)
                # update flb2
            else:
                flb3 = flb3
            if repair3 == 'OFF':
                bucket3 = 3
            else:
                print('7.3', flush=True)
                bucket3,position3,check_switch3,symboll3,repair3,order_switch3,tr3,p3s1,s13,p3s2,s23 = taskx(bucket3,position3,check_switch3,symboll3,repair3,order_switch3,tr3,p3s1,s13,p3s2,s23,flb3)
                # update  bucket3,position3,check_switch3,symboll3,repair3,order_switch3,tr3,p3s1,s13,p3s2,s23
                Data_array[0:1, 3:4] = bucket1
                Data_array[1:2, 3:4] = position1
                Data_array[2:3, 3:4] = check_switch1
                Data_array[3:4, 3:4] = repair1
                Data_array[4:5, 3:4] = tr1
                Data_array[5:6, 3:4] = p1s1
                Data_array[7:8, 3:4] = s11
                Data_array[8:9, 3:4] = p1s2
                Data_array[9:10, 3:4] = s21
                os.remove('Data_array.npy')
                np.save('Data_array', Data_array)

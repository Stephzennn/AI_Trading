# AI Trading Algorithm and BOT.
#### Video Demo:  <https://youtu.be/hN3vTbwxgMc>
#### Description:

This program is designed to manage a month-long trading operation using a machine learning model. The core concept behind this program is to leverage the predictive power of a machine learning model to construct a trading algorithm for an automated bot. This bot is capable of executing trades based on the predictions made by the model.

One of the primary challenges when using a machine learning model for trading is the model’s propensity to overfit the training dataset. Overfitting occurs when a model learns the training data too well, to the point where it performs poorly on new, unseen data. This is a common problem in machine learning and can lead to inaccurate predictions.

To mitigate this, one approach is to divide the training and testing datasets more evenly. However, this could potentially compromise the model’s performance, as it might not have enough data to learn effectively.


In this program, we tackle the overfitting issue in a unique way: we use it as a feature rather than a bug. The model is trained on the training dataset but is specifically optimized for the most recent 30 or 40 days of trading data. From all the trained models, we select the one that generated the most profit during the last 30 or 40 days of trading.

These chosen models are then further optimized by cross-validating them for different commodities within a predefined pool. After this optimization process, we select the two most successful models for a given commodity. These models are then combined using a Bayesian formula to produce the final prediction, which serves as a trading signal for the bot.

The models are refreshed every two weeks to ensure that the predictions are in tune with the current trends in the futures market. This approach effectively turns the overfitting problem into an advantage, as the models are continually updated to reflect recent market conditions.

Given the large size of the training data, we utilize the Google Colab server to train our models. The trading bot operates by loading the chosen training models, along with their respective scales, into a server. It then attempts to trade specific commodities around the clock, barring any systematic API failures.

The bot also incorporates a risk control system. Each individual commodity is assigned a specific risk tolerance level, and there is an overall risk parameter for the combination of commodities. These parameters can be adjusted according to our preference, allowing for customized performance and risk management.

In summary, this program represents a novel approach to algorithmic trading, leveraging machine learning models and turning potential drawbacks into advantages for more effective and profitable trading. It offers flexibility, customization, and robust risk management, making it a powerful tool for modern trading operations.

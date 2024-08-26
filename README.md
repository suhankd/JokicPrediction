# 
**Nikola Jokic Points Predictor**

A project I made to predict Nikola Jokic's points using previous seasons. I chose Jokic because of his relative consistency over the past 5 years, stable playstyle, low injury rate, and roster continuity with no major trades/signings, making it easier to identify and isolate trends within his performances.

Here are some other players I considered for this project, along with why I didn't choose them:

1. Kevin Durant : Team(s) too volatile.
2. SGA : Too young/Not enough data.
3. Steph Curry : Team is uncertain going forward.
4. Joel Embiid : Fraud.

BeautifulSoup was used to scrape performance data from [basketball-reference.com](http://basketball-reference.com/). I then used a temporal convolutional network for the model, feeding in different seasons one at a time, and then testing the model on the 2024 season.

Some notes :

1. I trained the model using season-by-season data batches rather than normalizing the entire dataset at once, due to the non-normal distribution of the data. Intuitively this seemed to work best.
2. Downloaded the source code for Steph Curry's 2016 season and used it as reference when scraping.
3. The Euclidean distance between the predictions and the y-Matrix were promising, but when the dynamic time warping (DTW) algorithm was applied and generated a warping curve, even stronger similarities between the predictions and the actual data were noticed. With a bit of human intuition added in, this model could be very effective
4. Will implement a random forest regressor and an LSTM-based model on this dataset as well.

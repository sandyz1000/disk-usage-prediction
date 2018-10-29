# Disk Storage Analysis

Disk usage analysis is very common need in data center, user will be informed before hand about his/her usage statistics and recommends if he or she need to buy more storage from the vendor so that once doesn't get any hindrance during the operation of the business.



#### Modelling Techniques

I have used three such modelling techniques and compares mean absolute of the three models and check the plot to see how the three models predict.

1. **Auto regressive Model**:  The dependent variable is regressed against one or more lagged values of itself the model is called **autoregressive**. I have used previous 6 months data as a features and try what will be the storage usage for the coming month

2. **Moving Average Model**: MA(q) models are very similar to AR(p) models. The difference is that the MA(q) model is a linear combination of past white noise error terms as opposed to a linear combo of past observations like the AR(p) model. The motivation for the MA model is that we can observe "shocks" in the error process directly by fitting a model to the error terms. 

3. **Random Walk Model**: A random walk is a time series model xt such that x(t)= x(t-1) + w(t), where w(t) is discreate white noise



    
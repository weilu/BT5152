# use bagging models to predict
predict_bagging_expected <- function(models, test_data) {

  prices <- sapply(models, function(m) {
    return(predict(m, test_data))
  })

  ### START CODE HERE ### (≈ 1 line)
  # `prices` is a matrix of predicted prices, where each row correspond to
  # a sample in `test_data`, and each column is a predicted price from one
  # of the models. We need to return the average predicted prices across
  # all models. You might find function `rowSums` useful.
  return()
  ### END CODE HERE ###
}

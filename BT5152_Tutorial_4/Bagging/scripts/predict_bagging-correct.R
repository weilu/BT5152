# use bagging models to predict
predict_bagging <- function(models, test_data) {
  prices <- sapply(models, function(m) {
    return(predict(m, test_data))
  })
  return(rowSums(prices) / length(models))
}

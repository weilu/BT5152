# use boosted models to predict
predict_boosting <- function(models, test_data) {
  # save all model predictions to a matrix
  # each row corresponds to a sample in test_data
  # each column corresponds to a model
  preds <- sapply(models, function(m) {
    return(predict(m, test_data, type='class'))
  })

  # apply majority voting
  voted_pred <- apply(preds, 1, function(row) {
    ### START CODE HERE ### (≈ 1 line)
    # find the class with the highest frequency in a row, extract the class name.
    # You might find the following function useful:
    # `table`, `which.max`, `names`
    return()
    ### END CODE HERE ###
  })

  # convert the string values in voted_pred to factors
  return(factor(voted_pred=='good', levels=c(T, F), labels=c('good', 'bad')))
}

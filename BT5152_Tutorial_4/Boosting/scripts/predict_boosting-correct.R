# use boosted models to predict
predict_boosting <- function(models, test_data) {
  preds <- sapply(models, function(m) {
    return(predict(m, test_data, type='class'))
  })
  # apply majority voting
  voted_pred <- apply(preds, 1, function(row) {
    return(names(which.max(table(row))))
  })
  return(factor(voted_pred=='good', levels=c(T, F), labels=c('good', 'bad')))
}

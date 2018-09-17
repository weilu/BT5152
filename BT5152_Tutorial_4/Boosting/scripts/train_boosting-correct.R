# build manual boosting models
train_boosting <- function(n_trees, training_data) {
  n_sample <- nrow(training_data)

  #start with equal weights
  weights <- as.numeric(unlist(list(rep(1/n_sample, n_sample))))

  models <- list()
  for(i in 1:n_trees){
    #sample with weights
    sample_indexes <- sample(1:n_sample, prob=weights, replace=TRUE)
    m <- rpart(Class ~., training_data[sample_indexes, ])
    models <- c(models, list(m))

    #calculate new probabilities based on prediction accuracy on training set
    actual_prob <- ifelse(as.character(training_data$Class) == 'good', 1, 0)
    pred_prob <- as.data.frame(predict(m, training_data, type='prob'))$good
    err <- abs(actual_prob - pred_prob)
    weights <- as.numeric(err/sum(err))
  }

  return(models)
}

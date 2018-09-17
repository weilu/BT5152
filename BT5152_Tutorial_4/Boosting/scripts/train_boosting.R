# build manual boosting models
train_boosting <- function(n_trees, training_data) {
  n_sample <- nrow(training_data)

  # start with equal weights
  weights <- as.numeric(unlist(list(rep(1/n_sample, n_sample))))

  models <- list()
  for(i in 1:n_trees){
    # sample with weights
    sample_indexes <- sample(1:n_sample, prob=weights, replace=TRUE)
    m <- rpart(Class ~., training_data[sample_indexes, ])
    models <- c(models, list(m))

    # calculate new weights based on prediction accuracy on the training set
    actual_prob <- ifelse(as.character(training_data$Class) == 'good', 1, 0)
    pred_prob <- as.data.frame(predict(m, training_data, type='prob'))$good

    # The idea is that the bigger the distance between actual and predicted
    # probability of a given sample, the more weights we want to assign to
    # it, so it's more likely to be selected in the next round of training.
    ### START CODE HERE ### (≈ 2 line)
    err <- 0 # calculate err as the absolute difference between actual_prob and pred_prob
    weights <- 0 # set weights to be the proportion of err of a single sample over sum of errors across all samples
    # note that both `err` and `weights` should be vectors, each element
    # correspond to a single sample in the training dataset
    ### END CODE HERE ###
  }

  return(models)
}


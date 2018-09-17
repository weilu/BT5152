# build manual bagging models
train_bagging <- function(n_bags, training_data){
  # create & return n_bags of rpart models
  models <- lapply(1:n_bags, function(i) {

    n_sample <- nrow(training_data)

    ### START CODE HERE ### (≈ 1 line)
    # invoke `sample` function here to sample n_sample rows with replacement
    # and assign the returned indexes to `sample_indexes`
    ### END CODE HERE ###

    new_training_data <- training_data[sample_indexes, ]
    return(rpart(price ~., new_training_data))
  })
  return(models)
}


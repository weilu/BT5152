# build manual bagging models
train_bagging <- function(n_bags, training_data){
  # create & return n_bags of rpart models
  models <- lapply(1:n_bags, function(i) {
    n_sample <- nrow(training_data)
    sample_indexes <- sample(1:n_sample, replace=TRUE)
    new_training_data <- training_data[sample_indexes, ]
    return(rpart(price ~., new_training_data))
  })
  return(models)
}

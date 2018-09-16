library(tidyverse)
library(ggplot2)
library(rpart)
library(Metrics)
library(adabag)

options(dplyr.width = Inf)

data('diamonds')

set.seed(1337)

diamonds <- as.tibble(diamonds)
diamonds$index <- 1:nrow(diamonds)

training <- withr::with_seed(42, sample_frac(diamonds, 0.8))
test <-diamonds %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)

model <- rpart(price ~., training)
out_train <- predict(model, training)
rpart_mae_train <- mae(training$price, out_train)
print(rpart_mae_train)
out <- predict(model, test)
rpart_mae <- mae(test$price, out)
print(rpart_mae)


boost <- function(n_trees, training_data) {
  n_sample <- nrow(training_data)

  #start with equal probabilities
  probs <- as.numeric(unlist(list(rep(1/n_sample, n_sample))))

  #for each tree
  for(i in 1:n_trees){

    #sample with weights
    s1 <- sample(1:n_sample, prob = probs, replace = TRUE)

    ti <- training_data[s1,]

    m <- rpart(price ~., ti)
    p <- predict(m, test)

    out = out + p

    #calculate new probabilities
    err <- abs(training_data$price - predict(m, training_data))
    probs <- as.numeric(err/sum(err))
  }

  out = out/n_trees
  return(out)
}

out_b <- boost(30, training)
boosted_mae <- mae(test$price, out_b)
print(boosted_mae)

# ada boosting only works for classification


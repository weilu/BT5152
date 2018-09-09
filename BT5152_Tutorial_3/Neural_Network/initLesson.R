library(tidyverse)
library(MASS)
library(Metrics)
library(neuralnet)

data('Boston')

set.seed(42)

boston <- as.tibble(Boston)
boston$index <- 1:nrow(Boston)

training <- withr::with_seed(42, sample_frac(boston, 0.8))
test <- boston %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)

# standardize training & test data
scaled_training <- training %>% mutate_all(scale)
scaled_test <- test %>% mutate_all(scale)

# create the formula field
col_names <- names(scaled_training)
formula <- as.formula(paste("medv ~", paste(col_names[!col_names %in% "medv"], collapse = " + ")))

# default parameters doesn't converge in time
model_nn <- neuralnet(formula, data=scaled_training)

# increase stepmax to give it more time to converge
model_nn <- neuralnet(formula, data=scaled_training, stepmax=1e6)
plot(model_nn)

# The compute() function works a bit differently from the predict() functions we've used so far.
# It returns a list with two components: $neurons, which stores the neurons for each layer in the network, and $net.result, which stores the predicted values.
results <- compute(model_nn, dplyr::select(scaled_test, -medv))
scaled_pred <- results$net.result

mae(scaled_test$medv, scaled_pred)

# 1 layer with 5 nodes
model_nn_1x5 <- neuralnet(formula, data=scaled_training, stepmax=1e6, hidden=5)
plot(model_nn_1x5)
results_1x5 <- compute(model_nn_1x5, dplyr::select(scaled_test, -medv))
scaled_pred_1x5 <- results_1x5$net.result
mae(scaled_test$medv, scaled_pred_1x5)

# 2 layers with total 5 nodes
model_nn_2x3 <- neuralnet(formula, data=scaled_training, stepmax=1e6, hidden=c(3, 2))
plot(model_nn_2x3)
results_2x3 <- compute(model_nn_2x3, dplyr::select(scaled_test, -medv))
scaled_pred_2x3 <- results_2x3$net.result
mae(scaled_test$medv, scaled_pred_2x3)

# use a different activation function
model_nn_tanh <- neuralnet(formula, data=scaled_training, stepmax=1e6, act.fct='tanh')
results_tanh <- compute(model_nn_tanh, dplyr::select(scaled_test, -medv))
scaled_pred_tanh <- results_tanh$net.result
mae(scaled_test$medv, scaled_pred_tanh)

# unscale predicted values
pred <- scaled_pred * attr(scaled_test$medv, 'scaled:scale') + attr(scaled_test$medv, 'scaled:center')

# visualize actual values vs. predicted values
plot(test$medv, pred)
abline(0, 1)


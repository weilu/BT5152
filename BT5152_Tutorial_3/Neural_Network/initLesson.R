library(tidyverse)
library(mlbench)
library(Metrics)
library(neuralnet)

options(dplyr.width = Inf)

data('BostonHousing')

set.seed(1337)

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

denormalize <- function(x, minx, maxx) {
  return(x * (maxx - minx) + minx)
}

boston <- as.tibble(BostonHousing) %>%
  mutate(chas = as.numeric(levels(chas)[chas]))
boston$index <- 1:nrow(boston)

training <- withr::with_seed(42, sample_frac(boston, 0.8))
test <- boston %>%
  anti_join(training, by = 'index') %>%
  select(-index)
training <- training %>% select(-index)

# standardize training & test data
scaled_training <- training %>% mutate_all(normalize)
scaled_test <- test %>% mutate_all(normalize)

# create the formula field
col_names <- names(scaled_training)
formula <- as.formula(paste("medv ~", paste(col_names[!col_names %in% "medv"], collapse = " + ")))

model_nn <- neuralnet(formula, data=scaled_training)
# plot(model_nn)

# The compute() function works a bit differently from the predict() functions we've used so far.
# It returns a list with two components: $neurons, which stores the neurons for each layer in the network, and $net.result, which stores the predicted values.
results <- compute(model_nn, select(scaled_test, -medv))
scaled_pred <- results$net.result

mae(scaled_test$medv, scaled_pred)

# 1 layer with 2 nodes
model_nn_2x1 <- neuralnet(formula, data=scaled_training, hidden=3)
# plot(model_nn_2x1)
results_2x1 <- compute(model_nn_2x1, select(scaled_test, -medv))
scaled_pred_2x1 <- results_2x1$net.result
mae(scaled_test$medv, scaled_pred_2x1)

# 2 layers with 3 nodes and 2 nodes each
model_nn_3x2 <- neuralnet(formula, data=scaled_training, hidden=c(3, 2))
# plot(model_nn_3x2)
results_3x2 <- compute(model_nn_3x2, select(scaled_test, -medv))
scaled_pred_3x2 <- results_3x2$net.result
mae(scaled_test$medv, scaled_pred_3x2)

# use a different activation function
model_nn_tanh <- neuralnet(formula, data=scaled_training, hidden=c(3, 2), act.fct='tanh')
results_tanh <- compute(model_nn_tanh, select(scaled_test, -medv))
scaled_pred_tanh <- results_tanh$net.result
mae(scaled_test$medv, scaled_pred_tanh)

# unscale predicted values
pred <- denormalize(scaled_pred, min(scaled_test$medv), max(scaled_test$medv))

# visualize actual values vs. predicted values
plot(test$medv, pred)
abline(0, 1)


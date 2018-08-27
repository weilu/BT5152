library(tidyverse)
data(iris)

set.seed(42)
iris_data <- as.tibble(iris)
iris_data$id <- 1:nrow(iris_data)

iris_train <- iris_data %>% sample_frac(2/3)
iris_test <- anti_join(iris_data, iris_train, by = 'id')

# remove id column from both datasets
iris_train <- iris_train %>% select(-id)
iris_test <- iris_test %>% select(-id)

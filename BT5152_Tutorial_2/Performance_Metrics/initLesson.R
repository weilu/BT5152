library(tidyverse)
library(mlbench)
library(C50)

data("BreastCancer")

breast_cancer <- as.tibble(BreastCancer)
train <- withr::with_seed(42, sample_frac(breast_cancer, 0.8))
test <- breast_cancer %>%
  anti_join(train, by = 'Id') %>% # side effect of anti_join: rows with NA removed
  select(-Id)
train <- train %>% select(-Id)

model_c50 <- C5.0(Class ~ ., data=train)


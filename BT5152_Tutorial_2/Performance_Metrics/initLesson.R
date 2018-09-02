library(tidyverse)
library(mlbench)
library(C50)
library(e1071)

data("BreastCancer")

breast_cancer <- as.tibble(BreastCancer)
train <- withr::with_seed(42, sample_frac(breast_cancer, 0.8))
test <- breast_cancer %>%
  anti_join(train, by = 'Id') %>% # side effect of anti_join: rows with NA removed
  select(-Id)
train <- train %>% select(-Id)

# prepare the decision classifier
model_c50 <- C5.0(Class ~ ., data=train)

# build a naive bayes classifer & prepare the roc plottable object for comparison purpose
model_nb <- naiveBayes(Class ~ ., data=train)
predictions_prob_nb <- predict(model_nb, test, type='raw')[, 2]
pred_nb <- prediction(predictions_prob_nb, labels=test$Class)
roc_nb <- performance(pred_nb, measure='tpr', x.measure='fpr')


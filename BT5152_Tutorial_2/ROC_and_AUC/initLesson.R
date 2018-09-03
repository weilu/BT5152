library(tidyverse)
library(mlbench)
library(C50)
library(e1071)

data("BreastCancer")

# remove any row with NA values
breast_cancer <- as.tibble(BreastCancer) %>%
  filter(complete.cases(.))

# Ids are not unique, so we create an index column
breast_cancer$index <- 1:nrow(breast_cancer)

training <- withr::with_seed(42, sample_frac(breast_cancer, 0.8))
test <- breast_cancer %>%
  anti_join(training, by = 'index') %>%
  select(-Id)
training <- training %>% select(-index)

# prepare the decision classifier
model_c50 <- C5.0(Class ~ ., data=training)

# build a naive bayes classifer & prepare the roc plottable object for comparison purpose
model_nb <- naiveBayes(Class ~ ., data=training)
predictions_prob_nb <- predict(model_nb, test, type='raw')[, 2]
pred_nb <- prediction(predictions_prob_nb, labels=test$Class)
roc_nb <- performance(pred_nb, measure='tpr', x.measure='fpr')


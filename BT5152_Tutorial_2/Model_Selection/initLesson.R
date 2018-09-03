library(tidyverse)
library(mlbench)
library(C50)
library(caret)

data("BreastCancer")

# remove any row with NA values
breast_cancer <- as.tibble(BreastCancer) %>%
  filter(complete.cases(.)) %>%
  select(-Id)

# Ids are not unique, so we create an index column
breast_cancer$index <- 1:nrow(breast_cancer)

training <- withr::with_seed(42, sample_frac(breast_cancer, 0.8))
test <- breast_cancer %>%
  anti_join(training, by = 'index') %>%
  select(-index)
training <- training %>% select(-index)

set.seed(1337)

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

boston <- as.tibble(BostonHousing) %>%
  mutate(chas = as.numeric(levels(chas)[chas]))
boston$index <- 1:nrow(boston)

training <- withr::with_seed(42, sample_frac(boston, 0.8))
test <- boston %>%
  anti_join(training, by = 'index') %>%
  dplyr::select(-index)
training <- training %>% dplyr::select(-index)


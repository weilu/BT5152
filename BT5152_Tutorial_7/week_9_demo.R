library(topicmodels)
library(ggplot2)
library(dplyr)
library(tidytext)

# Load AssociatePress data
data("AssociatedPress")

# Since LDA is an unsupervised algorithm, how do we know what is the optimal number of topics?
# We are going to separate the data to train and held-out dataset
# 90% of the data is used for training

train_idx <- sample(AssociatedPress$nrow * 0.9)

train <- AssociatedPress[train_idx,]
# Note that "test" is used very loosely here and this is not a test dataset
# test is the held-out dataset.
test <- AssociatedPress[-train_idx,]

# Training a single LDA model
# Note: LDA in `topicsmodels` does not take in a TF-IDF DocumentTermMatrix
# Technically, TF-IDF DocumentTermMatrix should yield better distributions
model <- LDA(train, k = 50, method = "Gibbs",
             control = list(verbose = 1, iter = 100))

# Calculate perplexity
# Since we are using the held-out dataset (test),
# we need to estimate the topic distributions (theta)
perplexity(model, test, use_theta = TRUE, estimate_theta = TRUE)

# Top 5 terms by topic
top_terms <- terms(model, 5)

# Top 5 topics by document
top_topics <- tbl_df(data.frame(t(topics(model, 5))))

# Getting the document-topics distribution for the held-out dataset
test_output <- posterior(model, test)


# Parallel Processing to train multiple LDA models
library(parallel)
library(doParallel)

# Create cluster to run R in parallel; best to use total number of CPU - 1
cl <- makePSOCKcluster(detectCores() - 1)

# Allow libraries such as doParallel and tm to access the cluster
registerDoParallel(cl)

# We are going to train models from 10 to 40 with in multiple of 5
ks <- seq(50, 200, 50)

# Use parSapply to pass in additional parameters to LDA
models <- parSapply(cl = cl, ks,
                    function(k, data) topicmodels::LDA(data, k = k, method = "Gibbs", control = list(iter = 100)),
                    data = train)

# We use the held-out dataset to compute perplexity
# Explain perplexity
perplexities <- parSapply(cl = cl, models,
                          function(m, data) topicmodels::perplexity(m, data, use_theta = TRUE, estimate_theta = TRUE),
                          data = test)

# Getting the index of the model with the lowest perplexity
optimal_idx <- which.min(perplexities)

# Let's plot how the perplexity varies over k
metrics_df <- tbl_df(data.frame(k = ks, perplexity = perplexities))

ggplot(metrics_df) + geom_line(aes(x = k, y = perplexity)) +
  ggtitle("Perplexity over number of topics")

# Now we use tidytext to extract the probability distribution in data frame

topics_terms_df <- tidy(models[[optimal_idx]], matrix = "beta")
documents_topics_df <- tidy(models[[optimal_idx]], matrix = "gamma")

# Plot the top terms in each topic
# Warning: if your number of topics is big, this is will very slow.
topics_terms_df %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# Top 5 terms by topic
top_terms <- terms(models[[optimal_idx]], 5)

# Top topics by document
top_topics <- tbl_df(data.frame(t(topics(models[[optimal_idx]], 5))))

test.topics <- posterior(models[[optimal_idx]], test)

stopCluster(cl)

# Reference: https://www.tidytextmining.com/topicmodeling.html

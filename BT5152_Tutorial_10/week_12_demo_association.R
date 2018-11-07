install.packages(c("arules", "arulesViz"))

library(arules)
library(arulesViz)
library(tidyverse)
library(reshape2)

# MovieLens Dataset

movie_lens_small_url <- "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
download.file(movie_lens_small_url, destfile = "data/ml-latest-small.zip")
unzip(zipfile = "data/ml-latest-small.zip")

# Load all the data as data frames
movies_df <- as_tibble(read_csv("data/ml-latest-small/movies.csv"))
links_df <- as_tibble(read_csv("data/ml-latest-small/links.csv"))
tags_df <- as_tibble(read_csv("data/ml-latest-small/tags.csv"))
ratings_df <- as_tibble(read_csv("data/ml-latest-small/ratings.csv"))

# Number of users
ratings_df %>% distinct(userId) %>% count()

# Number of movies
ratings_df %>% distinct(movieId) %>% count()

# Distribution of average rating by users
ratings_df %>% 
  group_by(userId) %>% 
  summarise(average_rating = mean(rating)) %>%
  ggplot(aes(average_rating)) + geom_histogram() + ggtitle("Average Rating Distribution for Users")

# Distribution of average rating by movies
ratings_df %>% 
  group_by(movieId) %>% 
  summarise(average_rating = mean(rating)) %>%
  ggplot(aes(average_rating)) + geom_histogram() + ggtitle("Average Rating Distribution for Movies")

# Highly-rated movies watched by each user
movies_watched <- ratings_df %>%
  filter(rating > 3) %>%
  group_by(userId) %>% 
  summarise(movies_watched = list(unique(movieId)), number_of_movies_watched = n()) %>% 
  arrange(desc(number_of_movies_watched))

head(movies_watched, 10)

# Convert to transaction for arules
# Basically it is a sparse matrix
transactions = as(movies_watched$movies_watched, "transactions")

# Check the dimension
dim(transactions@data)

# Plot a histogram to 
hist(size(transactions))

# Let's check the stats
summary(transactions)

# Check what is movieId 356
movies_df[movies_df$movieId == 318,]

# Number of users watched movieId 356
ratings_df[ratings_df$movieId == 356,]

# Plot the relative item frequency with minimum 0.1 support
itemFrequencyPlot(transactions, support = 0.1)

# Plot the absolute item frequency with minimum 100
itemFrequencyPlot(transactions, support = 100, type = "absolute")

# Top 20 movies
itemFrequencyPlot(transactions, topN = 20)

# Let's look at the matrix visually 
image(transactions)

# Run a apriori model with 0.1 as support and minimum confidence of 80% and the minimum number of movies watch to be 4
# Ideally you will want to recommend movies that are only rated highly, so for example, you can remove rows with rating < 3
rules <- apriori(data = transactions, parameter = list(support = 0.1, confidence = 0.8, minlen = 4))

# Look at the summary of the rules generated
summary(rules)

# How about the first 10 rules
inspect(rules[1:10])

# Let's make it friendly for data 
rules_df <- as.tibble(as(rules, "data.frame")) %>% 
  separate(rules, c("lhs", "rhs"), sep = "=>") %>%
  mutate(lhs = str_split(str_remove_all(lhs, "[\\{\\}]"), ",")) %>%
  mutate(rhs = str_remove_all(rhs, "[\\{\\}]"))

# You can write the rules generated to csv
# write(rules, file = "rules.csv", sep = ",")

# To make it useful by providing the names of the movies instead of the movieId
RULE_NO <- 1
movies_df[movies_df$movieId %in% c(as.numeric(unlist(rules_df$lhs[RULE_NO])), as.numeric(unlist(rules_df$rhs[RULE_NO]))),]
# Note that this doesn't preserve the order of the lhs and rhs, hence it appears duplicate

# You can change RULE_NO to see the different recommendation
movies_watched_df <- movies_df[movies_df$movieId %in% as.numeric(unlist(rules_df$lhs[RULE_NO])),] 
movie_to_recommend_df <- movies_df[movies_df$movieId %in% as.numeric(unlist(rules_df$rhs[RULE_NO])),]

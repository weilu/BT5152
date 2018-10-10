library(tibble)
library(dplyr)
library(parallel)
library(doParallel)
library(tm) # https://cran.r-project.org/web/packages/tm/tm.pdf
library(caret)

# Create cluster to run R in parallel; best to use total number of CPU - 1
cl <- makePSOCKcluster(detectCores() - 1)

# Allow libraries such as doParallel and tm to access the cluster
registerDoParallel(cl)
tm_parLapply_engine(cl)

# Read training data
train_neg <- VCorpus(DirSource("data/aclImdb/train/neg"))
train_pos <- VCorpus(DirSource("data/aclImdb/train/pos"))

# Set label with metadata
meta(train_neg, "LABEL", type = "indexed") <- "neg"
meta(train_pos, "LABEL", type = "indexed") <- "pos"

# Merge the corpus together
train <- c(train_neg, train_pos)

# Get the sentiment label
label <- as.factor(meta(train)$LABEL)

# Remove variable to save memory
rm("train_neg", "train_pos")

train <- tm_map(train, content_transformer(tolower))
train <- tm_map(train, removeWords, stopwords("english"))
train <- tm_map(train, removePunctuation)
train <- tm_map(train, stemDocument)
train <- tm_map(train, removeNumbers)

# Try creating DocumentTermMatrix (try without cl and with cl)
tf_dtm = DocumentTermMatrix(train, control = list(weighting = weightTf))

inspect(tf_dtm)

# Converting to matrix consumes a lot of memory
# So we are going to use high frequency words only
# For demo purpose, lowfreq is set high to quite to reduce
# dimensions so training time will decrease

freq_terms <- findFreqTerms(tf_dtm, lowfreq = 500)

# Recreate tfidf_dtm with freq_terms
# Note that this is a binary matrix
tf_dtm <- DocumentTermMatrix(train, 
                             control = list(dictionary = freq_terms, weighting = weightTf))

inspect(tf_dtm)

# Prepare features for text classifier

features_df <- data.frame(as.matrix(tf_dtm))

# Train a 10-fold CV model with Naives Bayes 
# Not going to run in class
control <- trainControl(method = 'cv', number = 10)
ptm <- proc.time()
model <- train(features_df, label, method = 'nb', trControl = control)
print(proc.time() - ptm)

pred <- predict(model$finalModel, features_df)

# Actually Need to do this on test data instead!

confusionMatrix(pred$class, label, positive = 'pos')

# Ngram tokenisation
library(RWeka)
BigramTokenizer = function(x) RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 2, max = 2))
tf_dtm = DocumentTermMatrix(train, 
                            control = list(tokenize = BigramTokenizer, weighting = weightTf))

gc()

inspect(tf_dtm)
# Stop the cluster when finish with the process
stopCluster(cl)

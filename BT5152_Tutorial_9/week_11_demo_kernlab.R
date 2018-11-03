library(kernlab)
library(caret)
library(ggplot2)
library(performanceEstimation) # This library provides function to calculate classification metrics such as micro and macro F1

# Loading the tutorial dataset
train <- read.csv("SVM_train.csv", colClasses = append(rep(c("numeric"), times = 2), "factor"))

# Splitting the first 1500 rows for training and the remaining as validation
train_t <- train[1:1500,]
train_v <- train[1501:2000,]

# Plot to see how data look like
ggplot(train_t, aes(x = x3, y = x2, color = y)) + geom_point(shape = 1) + ggtitle("training")
ggplot(train_v, aes(x = x3, y = x2, color = y)) + geom_point(shape = 1) + ggtitle("validation")

# Train a radial SVM model with cost of misclassification as 1 and gamma as 5
model.radial <- ksvm(y ~ ., data = train_t, kernel = 'rbfdot', C = 1, kpar = list(sigma = 5))

# Plot the kernel boundary and see how it is segment that 2 classes
plot(model.radial, data = train_t)

# Do prediction of the validation set
pred.radial = predict(model.radial, train_v)

# Compute the confusion matrix
cm <- confusionMatrix(pred.radial, train_v$y)

# Compute the classification metrics such as F scores
classificationMetrics(pred.radial, train_v$y)

# Unfortunately kernlab doesn't have built-in tuning functions
# You can use caret if you want to use kernlab


test_train_bagging <- function() {
  try({
    func <- get('train_bagging', globalenv())
    func_expected <- get('train_bagging_expected', globalenv())
    actual <- withr::with_seed(42, train_bagging(2, training))
    expected <- withr::with_seed(42, train_bagging_expected(2, training))
    t_length <- length(actual) == 2
    t_equal <- isTRUE(all.equal(actual, expected))
    ok <- all(t_length, t_equal)
  }, silent = TRUE)
  exists('ok') && isTRUE(ok)
}

test_predict_bagging <- function() {
  try({
    func <- get('predict_bagging', globalenv())
    func_expected <- get('predict_bagging_expected', globalenv())
    data = test[1:20, ]
    actual <- predict_bagging(models, data)
    expected <- predict_bagging_expected(models, data)
    t_length <- length(actual) == nrow(data)
    t_equal <- isTRUE(all.equal(actual, expected))
    ok <- all(t_length, t_equal)
  }, silent = TRUE)
  exists('ok') && isTRUE(ok)
}

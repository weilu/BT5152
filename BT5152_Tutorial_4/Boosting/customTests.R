test_train_boosting <- function() {
  try({
    func <- get('train_boosting', globalenv())
    func_expected <- get('train_boosting_expected', globalenv())
    actual <- withr::with_seed(42, train_boosting(2, training))
    expected <- withr::with_seed(42, train_boosting_expected(2, training))
    t_length <- length(actual) == 2
    t_equal <- isTRUE(all.equal(actual, expected))
    ok <- all(t_length, t_equal)
  }, silent = TRUE)
  exists('ok') && isTRUE(ok)
}

test_predict_boosting <- function() {
  try({
    func <- get('predict_boosting', globalenv())
    func_expected <- get('predict_boosting_expected', globalenv())
    data = test[1:20, ]
    actual <- predict_boosting(models, data)
    expected <- predict_boosting_expected(models, data)
    t_length <- length(actual) == nrow(data)
    t_equal <- isTRUE(all.equal(actual, expected))
    ok <- all(t_length, t_equal)
  }, silent = TRUE)
  exists('ok') && isTRUE(ok)
}

rpart_default = rpart(Species ~., data=iris_train)
best_cp <- rpart_default$cptable[which.min(rpart_default$cptable[,'xerror']), 'CP']
rpart_best_cp = rpart(Species ~., data=iris_train, control = rpart.control(cp=best_cp))

# hack to get around the bug:
# Error in rpartco(x) : no information available on parameters from previous call to plot()
plot(rpart_default)

par(mfcol = c(1, 2))

rpart.plot(rpart_default, main = "default: cp=0.01")
text(rpart_default, cex = 0.7)

rpart.plot(rpart_best_cp, main = "cp=best_cp")
text(rpart_best_cp, cex = 0.7)

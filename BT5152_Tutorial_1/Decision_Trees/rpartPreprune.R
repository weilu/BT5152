library(rpart)
library(rpart.plot)

rpart_default = rpart(Species ~., data=iris_train)
rpart_ms3 = rpart(Species ~., data=iris_train, control = rpart.control(minsplit = 3))
rpart_md2 = rpart(Species ~., data=iris_train, control = rpart.control(maxdepth = 2))
rpart_cp01 = rpart(Species ~., data=iris_train, control = rpart.control(cp = 0.5))

# hack to get around the bug:
# Error in rpartco(x) : no information available on parameters from previous call to plot()
plot(rpart_default)

par(mfcol = c(2, 2))

rpart.plot(rpart_default, main = "default: minsplit=20, maxdepth=30, cp=0.01")
text(rpart_default, cex = 0.7)

rpart.plot(rpart_ms3, main = "minsplit=3")
text(rpart_ms3, cex = 0.7)

rpart.plot(rpart_md2, main = "maxdepth=2")
text(rpart_md2, cex = 0.7)

rpart.plot(rpart_cp01, main = "cp=0.5")
text(rpart_cp01, cex = 0.7)

# Title     : TODO
# Objective : TODO
# Created by: ozanguldali
# Created on: 22.10.2020
set.seed(1)
n <- 199
m <- 50 # number of test obs
p <- 227
x <- matrix(rnorm(n*p), ncol=p)
y <- c(rep(1,100),rep(2,99))

x[y==1,1:10] <- x[y==1,1:10] + 2
x[y==2,11:20] <- x[y==2,11:20] - 2

xte <- matrix(rnorm(m*p), ncol=p)
yte <- c(rep(1,25),rep(2,25))

xte[yte==1, 1:10] <- xte[yte==1, 1:10] + 2
xte[yte==2, 11:20] <- xte[yte==2, 11:20] - 2
#print(x)

out <- TULIP::dsda.all(x=x, y=y, nfolds=5, lambda=c(1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20), alpha=1, eps=1e-7)

s <- out$s
yout <- out$pred
# print(out)
print(table(yout,yte))

test_size <- length(yte)
correct <- 0

for (i in 1:test_size) {
  if (yout[i] == yte[i]) {
    correct <- correct + 1
  }
}


function(x_tr, y_tr, x_ts=NULL, y_ts=NULL) {
    test_size <- length(y_ts)
    out <- TULIP::dsda.all(x=x, y=y, x.test.matrix=xte, y.test=yte, nfolds=5, lambda=c(1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20), alpha=1, eps=1e-7)
    pred <- out$pred
    for (i in 1:dim(pred)[2]) {
        y_pred <- as.vector()
        correct <- 0
        for (i in 1:test_size) {
            if (y_pred[i] == y_ts[i]) {
                correct <- correct + 1
            }
        }
        print("---")
        print(i)
        print("Test Success Ratio:", paste0(100 * correct / test_size, "%"), sep = " ")
        print("---")

    }
    return "done"
}
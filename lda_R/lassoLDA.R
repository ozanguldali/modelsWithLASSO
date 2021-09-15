# Title     : TODO
# Objective : TODO
# Created by: ozanguldali
# Created on: 12.07.2020
# source("/Users/ozanguldali/Documents/thesis/modelsWithLASSO/lda/penalizedLDA.R", local = TRUE)
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

# out <- penalizedLDA::PenalizedLDA(x,y,l=.14,K=1)
cv <- 5
folds <- list()

nrow <- dim(x)[1]
ncol <- dim(x)[2]
random_rows <- sample(1:nrow,rep=FALSE)

rounded_fold_size <- as.integer(nrow / cv)

residual_size <- nrow - rounded_fold_size*cv

initial <- 1
for (i in 1:cv) {
  final <- initial+rounded_fold_size-1
  if (residual_size > 0) {
    final <- final + 1
    residual_size <- residual_size - 1
  }

  folds <- append(folds, list(random_rows[initial:final]))

  initial <- final + 1
}




kf <- caret::createFolds(y, k=5)

for (i in  1:length(kf)) {
  fold_test <- kf[[i]]
  nrow_test <- length(fold_test)

  # fold_train <- vector(mode = integer, length = dim(x)[1] - nrow_test)
  fold_train <- NULL
  for (k in 1:length(kf)) {
    if (k != i) {
      fold_train <- c(fold_train, kf[[k]])
    }
  }
  nrow_train <- length(fold_train)



  x_test <- matrix(, nrow = nrow_test, ncol = ncol)
  y_test <- vector(, length = nrow_test)

  for (j in 1:nrow_test) {
    x_test[j, ] <- x[fold_test[j], ]
    y_test[j] <- y[j]
  }

  x_train <- matrix(, nrow = nrow_train, ncol = ncol)
  y_train <- vector(, length = nrow_train)

  for (j in 1:nrow_train) {
    x_train[j, ] <- x[fold_train[j], ]
    y_train[j] <- y[j]
  }

}

cv.out <- penalizedLDA::PenalizedLDA.cv(x,y,lambdas=c(1e-4,1e-3),nfold=5)
print(cv.out)
plot(cv.out)

out <- penalizedLDA::PenalizedLDA(x,y,xte=xte,K=1)

yout <- out$ypred[,out$K]

print(out)
print(table(yout,yte))

test_size <- length(yte)
correct <- 0

for (i in 1:test_size) {
  if (yout[i] == yte[i]) {
    correct <- correct + 1
  }
}

print(100 * correct / test_size)


function(x_tr, y_tr, x_ts=NULL, y_ts=NULL, cv) {

                    if (cv == 0) {

                        out <- penalizedLDA::PenalizedLDA(x=x_tr,y=y_tr,xte=x_ts, K=length(unique(y_tr))-1)
                        y_pred <- out$ypred[,out$K]
                        print(out)
                        print(table(y_pred,y_ts))

                        test_size <- length(y_ts)
                        correct <- 0
                        for (i in 1:test_size) {
                            if (y_pred[i] == y_ts[i]) {
                               correct <- correct + 1
                            }
                        }

                        print_value <- paste("Test Success Ratio:", paste0(100 * correct / test_size, "%"), sep = " ")
                        return(print_value)

                    } else {
                        set.seed(1)
                        success_ratio <- NULL
                        print("all is fine")

                        # kf <- caret::createFolds(y, k=cv)

                        nrow <- dim(x_tr)[1]
                        ncol <- dim(x_tr)[2]

                        random_rows <- sample(1:nrow,rep=FALSE)

                        rounded_fold_size <- as.integer(nrow / cv)
                        residual_size <- nrow - rounded_fold_size*cv

                        folds <- list()

                        initial <- 1
                        for (i in 1:cv) {
                            final <- initial+rounded_fold_size-1
                            if (residual_size > 0) {
                                final <- final + 1
                                residual_size <- residual_size - 1
                            }

                            folds <- append(folds, list(random_rows[initial:final]))

                            initial <- final + 1
                        }

                        for (i in  1:length(folds)) {
                            fold_test <- folds[[i]]
                            nrow_test <- length(fold_test)

                            # fold_train <- vector(mode = integer, length = dim(x)[1] - nrow_test)
                            fold_train <- NULL
                            for (k in 1:length(folds)) {
                                if (k != i) {
                                    fold_train <- c(fold_train, folds[[k]])
                                }
                            }
                            nrow_train <- length(fold_train)

                            x_test <- matrix(, nrow = nrow_test, ncol = ncol)
                            y_test <- vector(, length = nrow_test)

                            for (j in 1:nrow_test) {
                                x_test[j, ] <- x[fold_test[j], ]
                                y_test[j] <- y[j]
                            }

                            x_train <- matrix(, nrow = nrow_train, ncol = ncol)
                            y_train <- vector(, length = nrow_train)

                            for (j in 1:nrow_train) {
                                x_train[j, ] <- x[fold_train[j], ]
                                y_train[j] <- y[j]
                            }

                            out <- penalizedLDA::PenalizedLDA(x=x_train,y=y_train,xte=x_test,lambda=0.0001, K=length(unique(y_train))-1)

                            y_pred <- out$ypred[,out$K]
                            # print(out)
                            # print(table(y_pred,y_test))

                            correct <- 0
                            for (i in 1:nrow_test) {
                                print(y_pred[i])
                                print(y_test[i])
                                if (y_pred[i] == y_test[i]) {
                                   correct <- correct + 1
                                }
                            }

                            success_ratio <- c(success_ratio, 100 * correct / nrow_test)
                            print(paste0(cv, paste("-Fold CV -- Iteration", paste(i, "Test Success Ratio:", paste0(100 * correct / nrow_test, "%"), sep = " "), sep = " ")))

                        }

                        print_value <- paste0(cv, paste("-Fold CV Average Test Success Ratio:", paste0(mean(success_ratio), "%"), sep = " "))
                        return(print_value)

                    }

                }
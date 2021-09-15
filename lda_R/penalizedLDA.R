#PenalizedLDA <- function(x, y, xte = NULL, type = "standard", l, K = 2, chrom = NULL, lambda2 = NULL, standardized = FALSE, wcsd.x = NULL, ymat = NULL, maxiter = 20, trace = FALSE) {
#  if (sum(1 : length(unique(y)) != sort(unique(y))) > 0) stop("y must be a numeric vector, with values as follows: 1, 2, ....")
#  if (sum(is.na(x)) > 0 || sum(is.na(y)) > 0 || (!is.null(xte) && sum(is.na(xte)) > 0)) stop("No missing values allowed!!!")
#  if (K >= length(unique(y))) stop("Can have at most K-1 components of K unique classes")
#  yclass <- y
#  y <- ymat
#  if (is.null(ymat)) y <- MakeYMat(yclass)
#  if (type == "ordered" && is.null(lambda2)) stop("For type \'ordered\', lambda2 must be specified.")
#  xorig <- x
#
#  if (!standardized) {
#    if (is.null(wcsd.x)) {
#      if (length(y) <= 200) {
#        wcsd.x <- wcsd.matrix(x, y)
#      } else {
#        wcsd.x <- apply(x, 2, wcsd, yclass)
#      }
#      if (min(wcsd.x) == 0) stop("Some features have 0 within-class standard deviation.")
#    }
#    if (!is.null(xte)) xte <- scale(xte, center = apply(x, 2, mean), scale = wcsd.x)
#    x <- scale(x, T, scale = wcsd.x)
#  }
#  sqrt.sigma.bet <- t(scale(y, F, sqrt(apply(y, 2, sum)))) %*% x / sqrt(nrow(x))
#  while (sum(is.na(sqrt.sigma.bet)) > 0) {
#    sqrt.sigma.bet <- t(scale(y, F, sqrt(apply(y, 2, sum)))) %*% x / sqrt(nrow(x))
#    cat("retrying", fill = TRUE)
#  }
#  penpca <- PenalizedPCA(x = sqrt.sigma.bet, lambda = l, K = K, type = type, chrom = chrom, lambda2 = lambda2, maxiter = maxiter, trace = trace)
#  Usparse <- penpca$v
#  if (K == 1) Usparse <- matrix(Usparse, ncol = 1)
#  if (sum(is.na(Usparse)) > 0) {
#    Usparse[is.na(Usparse)] <- 0
#    #Usparse <- matrix(Usparse,ncol=K)
#    if (K == 1) Usparse <- matrix(Usparse, ncol = 1)
#  }
#  xtranssparse <- x %*% Usparse
#  if (K == 1) xtranssparse <- matrix(xtranssparse, ncol = 1)
#  if (!is.null(xte)) {
#    #xtetranssparse <- matrix(xte%*%Usparse, ncol=K)
#    xtetranssparse <- xte %*% Usparse
#    if (K == 1) xtetranssparse <- matrix(xtetranssparse, ncol = 1)
#    ypredsparsemat <- matrix(NA, ncol = K, nrow = nrow(xte))
#    for (k in 1 : K) {
#      ypredsparsemat[, k] <- Classify(matrix(xtranssparse[, 1 : k], ncol = k), matrix(xtetranssparse[, 1 : k], ncol = k), yclass)
#    }
#    obj <- (list(ypred = ypredsparsemat, discrim = Usparse, xproj = xtranssparse, xteproj = xtetranssparse, K = K, crits = penpca$crits, type = type, lambda = l, lambda2 = lambda2, wcsd.x = wcsd.x, x = xorig, y = yclass))
#    class(obj) <- "penlda"
#    return(obj)
#  }
#  obj <- (list(discrim = Usparse, xproj = xtranssparse, K = K, crits = penpca$crits, type = type, lambda = l, lambda2 = lambda2, wcsd.x = wcsd.x, x = xorig, y = yclass))
#  class(obj) <- "penlda"
#  return(obj)
#}
#
#predict.penlda <- function(object,xte,...){
#  # First need to standardize the training and test data sets as needed.
#  meanvec <- apply(object$x,2,mean)
#  if(is.null(object$wcsd.x)){
#    xte <- scale(xte,center=meanvec,scale=FALSE)
#    x <- scale(object$x,center=meanvec,scale=FALSE)
#  }
#  if(!is.null(object$wcsd.x)){
#    xte <- scale(xte,center=meanvec,scale=object$wcsd.x)
#    x <- scale(object$x, center=meanvec,scale=object$wcsd.x)
#  }
#  # Done standardizing.
#  # Now perform classification.
##  if(object$K==train){
##    ypred <- Classify(matrix(x%*%object$discrim,ncol=train), matrix(xte%*%object$discrim,ncol=train), object$y)
##    return(list(ypred=ypred))
##  }
#  ypredsparsemat <- matrix(NA, nrow=nrow(xte), ncol=object$K)
#  for(k in 1:object$K){
#    ypredsparsemat[,k] <- Classify(matrix(x%*%object$discrim[,1:k],ncol=k), matrix(xte%*%object$discrim[,1:k],ncol=k), object$y)
#  }
#  return(list(ypred=ypredsparsemat))
#}
#
#plot.penlda <- function(x,...){
#  K <- x$K
#  par(mfrow=c(1,K))
#  for(k in 1:K) plot(x$discrim[,k], main=paste("Discriminant ", k, sep=""),xlab="Feature Index", ylab="")
#}
#
#print.penlda <- function(x,...){
#  cat("Number of discriminant vectors: ", x$K, fill=TRUE)
#  K <- x$K
#  for(k in 1:K){
#    cat("Number of nonzero features in discriminant vector ", k, ":", sum(x$discrim[,k]!=0),fill=TRUE)
#  }
#  if(K>1) cat("Total number of nonzero features: ", sum(apply(x$discrim!=0, 1, sum)!=0),fill=TRUE)
#  cat(fill=TRUE)
#  cat("Details:", fill=TRUE)
#  cat("Type: ", x$type, fill=TRUE)
#  if(x$type=="standard") cat("Lambda: ", x$lambda, fill=TRUE)
#  if(x$type=="ordered"){
#    cat("Lambda: ", x$lambda,fill=TRUE)
#    cat("Lambda2: ", x$lambda2, fill=TRUE)
#  }
#}
#
#library(flsa)
#
#permute.rows <- function(x){
#    dd <- dim(x)
#      n <- dd[1]
#      p <- dd[2]
#      mm <- runif(length(x)) + rep(seq(n) * 10, rep(p, n))
#      matrix(t(x)[order(mm)], n, p, byrow = T)
#  }
#
#
#balanced.folds <- function(y, nfolds = min(min(table(y)), 10)){
#    totals <- table(y)
#      fmax <- max(totals)
#      nfolds <- min(nfolds, fmax)
#      # makes no sense to have more folds than the max class size
#      folds <- as.list(seq(nfolds))
#      yids <- split(seq(y), y)
#      # nice way to get the ids in a list, split by class
#      ###Make a big matrix, with enough rows to get in all the folds per class
#      bigmat <- matrix(NA, ceiling(fmax/nfolds) * nfolds, length(totals))
#      for(i in seq(totals)) {
#            bigmat[seq(totals[i]), i] <- sample(yids[[i]])
#          }
#      smallmat <- matrix(bigmat, nrow = nfolds) # reshape the matrix
#      ### Now do a clever sort to mix up the NAs
#      smallmat <- permute.rows(t(smallmat)) ### Now a clever unlisting
#      x <- apply(smallmat, 2, function(x) x[!is.na(x)])
#      if(is.matrix(x)){
#            xlist <- list()
#                for(i in 1:ncol(x)){
#                        xlist[[i]] <- x[,i]
#                      }
#                return(xlist)
#          }
#      return(x)
#  }
#
#
#
#
#
#
#MakeYMat <- function(y){
#  return(diag(length(unique(y)))[y,])
#}
#
##MakeYMat <- function(y){
##  ymat <- matrix(0, nrow=length(y), ncol=length(unique(y)))
##  for(i in train:ncol(ymat)) ymat[y==i,i] <- train
##  return(ymat)
##}
#
#MakeMeanVecs <- function(x,y){
#  Y <- MakeYMat(y)
#  return(solve(msqrt(t(Y)%*%Y))%*%t(Y)%*%x)
#}
#
#msqrt <- function(mat){
#  if(sum((mat-t(mat))^2)>1e-8) stop("Msqrt function only works if mat is symmetric....")
#  redo <- TRUE
#  while(redo){
#    eigenmat <- eigen(mat)
#    d <- eigenmat$values
#    d[abs(d)<(1e-12)] <- 0
#    a <- eigenmat$vectors%*%diag(sqrt(d))%*%t(eigenmat$vectors)
#    if(sum(is.na(a))==0) redo <- FALSE
#    if(redo) print('did one loop')
#  }
#  return(a)
#}
#
#soft <- function(mat,lam){
#  return(sign(mat)*pmax(abs(mat)-lam, 0))
#}
#
#
#Penalty <- function(v,lambda,type,chrom, lambda2){
#  if(type=="standard") return(lambda*sum(abs(v)))
#  if(type=="ordered"){
#    tots <- lambda*sum(abs(v))
#    for(chr in sort(unique(chrom))){
#      tots <- tots+lambda2*sum(abs(diff(v[chrom==chr])))
#    }
#    return(tots)
#  }
#}
#
#
#PenalizedPCACrit <- function(x, P, v, lambda, d, type, chrom, lambda2){
#  return(t(v)%*%t(x)%*%P%*%x%*%v-d*Penalty(v, lambda, type, chrom, lambda2))
#}
#
#
#PenalizedPCA <- function(x, lambda, K, type="standard",  chrom=NULL, lambda2=NULL, maxiter=30, trace=FALSE){
#  # Notice that this K is the number of components desired, NOT the number of classes in the classification problem.
#  # Here, x is (# of classes) \times p
#
#  # The criterion is maximize_b (b' Sigmabet b) - P(b) s.t. b' Sigmawit b = train
#  # Where Sigmawit=I and where P(b) = lambda||b||_1 or P(b) = lambda||b||_1 + lambda2 ||b_i - b_{i-train}||_1
#  # We take a MINORIZATION approach to this problem.
#  if(type=="ordered" && is.null(chrom)) chrom <- rep(1, ncol(x))
#  if(is.null(lambda2)) lambda2 <- lambda
#  crits <-  NULL
#  betas <- matrix(0, nrow=ncol(x), ncol=K)
#  critslist <- list()
#  for(k in 1:K){
#    if(trace) cat("Starting on component ", k, fill=TRUE)
#    if(k>1){
#      svda <- svd(x%*%betas)
#      u <- svda$u[,svda$d>(1e-10)]
#      P <- diag(nrow(x)) - u%*%t(u)
#    }
#    if(k==1) P <- diag(nrow(x))
#    svdx <- svd(t(x)%*%P)
#    d <- svdx$d[1]^2
#    beta <- svdx$u[,1]
#    crits <- c(crits, PenalizedPCACrit(x, P, beta, lambda, d, type, chrom=chrom, lambda2))
#    for(iter in 1:maxiter){
#      if((length(crits)<4 || abs(crits[length(crits)]-crits[length(crits)-1])/max(1e-3, crits[length(crits)]) > (1e-6)) && sum(abs(beta))>0){
#        if(trace) cat(iter,fill=FALSE)
#        tmp <- (t(x)%*%P)%*%(x%*%beta)
#        if(type=="standard") beta <- soft(tmp, d*lambda/2)
#        if(type=="ordered"){
#          for(chr in sort(unique(chrom))){
#            beta[chrom==chr] <- as.numeric(flsa(tmp[chrom==chr],  d*lambda/2,  d*lambda2/2))
#          }
#        }
#        beta <- beta/l2n(beta)
#        beta[is.na(beta)] <- 0
#        crits <- c(crits, PenalizedPCACrit(x, P, beta, lambda, d, type, chrom=chrom, lambda2))
#      }
#    }
#    if(trace) cat(fill=TRUE)
#    betas[,k] <- beta#cbind(betas, beta)
#    critslist[[k]] <- crits
#    if(min(diff(crits))<(-1e-6)) stop("min diff crits is too small!!!")
#    crits <- NULL
#  }
#  return(list(v=betas, crits=as.vector(critslist)))
#}
#
#
#
#
#
#l2n <- function(vec){
#  return(sqrt(sum(vec^2)))
#}
#
#
#
#
#
#
#diag.disc <-function(x, centroids, prior) {
#  dd <- t(x) %*% centroids
#  dd0 <- (rep(1, nrow(centroids)) %*% (centroids^2))/2 - log(prior)
#  scale(dd, as.numeric(dd0), FALSE) # this is -.5*||x_i - mu_k||^2+log(pi_k)
#}
#
#
#
#
#Classify <- function(xtr,xte,ytr,equalpriors=FALSE){ # I introduced unequal priors on 02/22/2010
#  prior <- rep(1/length(unique(ytr)), length(unique(ytr)))
#  if(!equalpriors){
#    for(k in 1:length(unique(ytr))) prior[k] <- mean(ytr==k)
#  }
#  # classify test obs to nearest training centroid.
#  if(is.matrix(xtr) && ncol(xtr)>1){
#    mus <- matrix(0, nrow=ncol(xtr), ncol=length(unique(ytr)))
#    for(k in 1:length(unique(ytr))){
#       mus[,k] <- apply(xtr[ytr==k,], 2, mean)
#     }
#  } else {
#    mus <- matrix(NA, nrow=1, ncol=length(unique(ytr)))
#    for(k in 1:length(unique(ytr))) mus[1,k] <- mean(xtr[ytr==k])
#  }
#  negdists <- diag.disc(t(xte), mus, prior)
#  return(apply(negdists,1,which.max))
#}
#
#
#
#
#
#wcsd <- function(vec, y){
#  K <- length(unique(y))
#  n <- length(vec)
#  tots <- 0
#  for(k in unique(y)){
#    tots <- tots + sum((vec[y==k]-mean(vec[y==k]))^2)
#  }
#  return(sqrt(tots/n))
#}
#
#
#wcsd.matrix <- function(x,Y){
#  n <- nrow(x)
#  return(sqrt((1/n)*apply(((diag(n)-Y%*%diag(1/apply(Y,2,sum))%*%t(Y))%*%x)^2,2,sum)))
#}

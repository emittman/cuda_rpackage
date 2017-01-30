context("Sum of squared errors")

set.seed(13017)

for(i in 1:3){
  K <- as.integer(4)
  reps <- 4
  V <- 2
  n_per_v <- 5
  group <- rep(1:K, each=reps)

  zeta <- sample(rep(0:(K-1), each=reps), K*reps)

  X <- kronecker(diag(V), rep(1, n_per_v))

  beta <- matrix(seq(1, 5, length.out = V*K), V, K)

  y <- t(matrix(rnorm(K*V*n_per_v*reps, X %*% beta[,group], .1), V*n_per_v, K*reps))

  data <- formatData(y, X, transform_y = identity)

  sse <- .Call("RsumSqErr", data, zeta, K, beta)

  bxxb <- sapply(1:K, function(k) reps * t(beta[,k]) %*% matrix(data$xtx,V,V) %*% beta[,k])
  yxb <- sapply(1:K, function(k) t(beta[,k]) %*% rowSums(data$xty[,which(zeta == k-1)]))
  yty <- sapply(1:K, function(k) sum(data$yty[which(zeta == k-1)]))

  sseR <- bxxb + yty - 2*yxb

  test_that("Answers comport",{
    expect_equal(sse, sseR)
  })
}
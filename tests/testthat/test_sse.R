context("Sum of squared errors")

set.seed(13017)

for(i in 1:3){
  K <- as.integer(20)
  G <- 20
  V <- 2
  n_per_v <- 5
  # group <- rep(1:V, each=reps)

  zeta <- sample(0:(K-1), G, replace=T)

  X <- kronecker(diag(V), rep(1, n_per_v))

  beta <- matrix(rnorm(K*V), V, K)

  y <- t(matrix(rnorm(G*V*n_per_v, X %*% beta[,zeta+1], .1), V*n_per_v, G))

  data <- formatData(y, X, transform_y = identity)

  sse <- .Call("RsumSqErr", data, zeta, K, beta)

  k_occ <- as.numeric(names(table(zeta)))+1
  Mk <- sapply(1:K, function(k) sum(zeta == k-1))
  
  bxxb <- sapply(k_occ, function(k) Mk[k] * t(beta[,k]) %*% matrix(data$xtx,V,V) %*% beta[,k])
  yxb <- sapply(k_occ, function(k) t(beta[,k]) %*% rowSums(data$xty[,which(zeta == k-1)]))
  yty <- sapply(k_occ, function(k) sum(data$yty[which(zeta == k-1)]))

  sseR <- bxxb + yty - 2*yxb

  test_that("Answers comport",{
    expect_equal(sse, sseR)
  })
}
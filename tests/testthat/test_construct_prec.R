context("Testing construct precision")

dim <- c(1,10)

for(d in dim){
  clusts <- 500
  x <- matrix(round(rnorm(d^2)*3,  1), d, d)
  xtx <- t(x) %*% x
  
  Mk <- rpois(clusts, 3) + 1
  lambda <- rlnorm(d)
  tau <- rlnorm(clusts)
  
  prec <- rep(xtx, times=clusts)
  len <- d*d
  dim(prec) <- c(d, d, clusts)
  
  Rprec <- sapply(1:clusts, function(cl){
    submat <- prec[,,cl] * Mk[cl]
    if(d == 1){
      submat <- submat + lambda/tau[cl]
    } else{
      diag(submat) <- diag(submat) + lambda/tau[cl]
    }
    submat
  })
  
  dim(Rprec) <- c(d, d, clusts)
  
  
  Cprec <- Rconstruct_prec(xtx, Mk, lambda, tau, clusts, d)
  
  test_that("Correct values",{
    expect_equal(Rprec, Cprec)
  })
}
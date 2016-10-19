context("Testing construct precision")

dim <- 2
clusts <- 2
x <- matrix(round(rnorm(dim^2)*3,  1), dim, dim)
xtx <- t(x) %*% x

Mk <- rpois(clusts, 3) + 1
lambda <- rlnorm(dim)
tau <- rlnorm(clusts)

prec <- rep(xtx, times=clusts)
len <- dim*dim
dim(prec) <- c(dim, dim, clusts)

Rprec <- sapply(1:clusts, function(cl){
  submat <- prec[,,cl] * Mk[cl]
  diag(submat) <- diag(submat) + lambda/tau[cl]
  submat
})

dim(Rprec) <- c(dim, dim, clusts)


Cprec <- Rconstruct_prec(xtx, Mk, lambda, tau, clusts, dim)

test_that("Correct values",{
  expect_equal(Rprec, Cprec)
})
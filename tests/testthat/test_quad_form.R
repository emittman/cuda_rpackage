context("Testing calculate quadratic form")

set.seed(2132510)

dim <- 6
n <- 100

Asqrt <- matrix(rnorm(dim*dim), dim, dim)
A <- t(Asqrt)%*%Asqrt
x <- matrix(rnorm(dim*n), dim, n)

resultR <- apply(x, 2, function(xj) t(xj) %*% A %*% xj)

resultC <- Rquadform_multipleK(A, as.numeric(x), n, dim)

test_that("Results are equiv.", {
  expect_equal(resultR, resultC)
})
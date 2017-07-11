context("Testing calculate quadratic form")

set.seed(2132510)

dim <- 6
n <- 100

Asqrt <- matrix(rnorm(dim*dim), dim, dim)
A <- t(Asqrt)%*%Asqrt
x <- matrix(rnorm(dim*n), dim, n)

resultR_K <- apply(x, 2, function(xj) t(xj) %*% A %*% xj)

resultC_K <- Rquadform_multipleK(A, as.numeric(x), n, dim)

test_that("multipleK", {
  expect_equal(resultR_K, resultC_K)
})

G <- 50

A <- sapply(1:G, function(g){
  Asqrt <- matrix(rnorm(dim*dim), dim, dim)
  t(Asqrt) %*% Asqrt
})
resultsR_GK <- as.numeric(sapply(1:G, function(g){
  xtA <- t(x) %*% matrix(A[,g], dim, dim)
  sapply(1:n, function(k){
    xtA[k,] %*% x[,k]
  })
}))
  
resultC_GK <- .Call("Rquadform_multipleGK", as.numeric(A),
                    as.numeric(x), as.integer(G), as.integer(n), as.integer(dim))

test_that("multipleGK", {
  expect_equal(resultR_GK, resultC_GK)
})
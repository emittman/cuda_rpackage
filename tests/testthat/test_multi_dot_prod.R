context("Dot products")

set.seed(12717)

dim <- as.integer(5)
n <- as.integer(1e4)

x <- matrix(rnorm(dim*n, 0, 100), dim, n)
y <- matrix(rnorm(dim*n, 0, .5), dim, n)

z <- .Call("Rmulti_dot_prod", x, y, dim, n)

Rz <- sapply(1:n, function(i) x[,i] %*% y[,i])

test_that("Correct output", {
  expect_equal(z, Rz)
})
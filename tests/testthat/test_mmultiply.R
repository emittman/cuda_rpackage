context("Device matrix multiply")

a1 <- 5
a2 <- 100
b1 <- 5
b2 <- 1000

A <- matrix(rnorm(a1*a2), a1, a2)
B <- matrix(rnorm(b1*b2), b1, b2)

outC <- Rdevice_mmultiply(A, B)

outR <- t(A) %*% B

test_that("Results are equiv.", {
  expect_equal(outR, outC)
})
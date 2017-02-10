context("Update means")

G <- 100
K <- 20
V <- 4

beta <- rnorm(K*V)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rgamma(K, 2, 2)
zeta <- as.integer(sample(0:(K-1), G, replace=T))

C <- matrix(c(1, 0, 0, 0,
              0, 1, 0, 0), 2, 4, byrow=2)

probs <- rbeta(G, 1, 1)
means <- rnorm(G*V)
meansquares <- rnorm(G*V)^2

chain <- formatChain(beta, pi, tau2, zeta, C, probs, means, meansquares)
step <- 12
outR <- .Call("Rtest_update_means", chain, as.integer(step))
means <- means + (beta-means)/step
meansquares <- meansquares + (beta^2 - meansquares)/step
probs <- probs + (colMeans(C %*% matrix(beta, V, K)>0) - probs)/step

test_that("probs equal",{
  expect_equal(outC[[1]], probs)
})

test_that("means equal",{
  expect_equal(outC[[2]], means)
})

test_that("meansquares equal",{
  expect_equal(outC[[3]], meansquares)
})

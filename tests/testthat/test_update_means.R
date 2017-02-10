context("Update means")

G <- 50
K <- 20
V <- 2

beta <- matrix(rnorm(K*V),V,K)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rgamma(K, 2, 2)
zeta <- as.integer(sample(0:(K-1), G, replace=T))

C <- matrix(c(1, 0,
              0, 1), 2, 2, byrow=2)

probs <- rbeta(G, 1, 1)
means <- matrix(rnorm(G*V),V,G)
meansquares <- matrix(rnorm(G*V)^2,V,G)

chain <- formatChain(beta, pi, tau2, zeta, C, probs, means, meansquares)
step <- 10
outC <- .Call("Rtest_update_means", chain, as.integer(step))
means <- means + (beta[,zeta+1] - means)/step
meansquares <- meansquares + ((beta[,zeta+1])^2 - meansquares)/step
probs <- probs + (apply(C %*% beta[,zeta+1] > 0, 2, min) - probs)/step

test_that("probs equal",{
  expect_equal(outC[[1]], as.numeric(probs))
})

test_that("means equal",{
  expect_equal(outC[[2]], as.numeric(means))
})

test_that("meansquares equal",{
  expect_equal(outC[[3]], as.numeric(meansquares))
})

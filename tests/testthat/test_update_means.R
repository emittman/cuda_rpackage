context("Update means")

G <- 50
K <- 20
V <- 3

beta <- matrix(rnorm(K*V),V,K)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rgamma(K, 2, 2)
zeta <- as.integer(sample(0:(K-1), G, replace=T))
alpha <- 1

C <- list(
  matrix(c(1, -1, 0,
           0, 1, -1), 2, 3, byrow=T),
  diag(3),
  matrix(c(1, 0, -1,
           .5, .5, -1,
           1, -.5, -.5), 3, 3, byrow=T)
)

probs <- matrix(rbeta(G*3, 1, 1), 3, G)
means <- matrix(rnorm(G*V),V,G)
meansquares <- matrix(rnorm(G*V)^2,V,G)

chain <- formatChain(beta, pi, tau2, zeta, alpha, C, probs, means, meansquares)
step <- 10
outC <- .Call("Rtest_update_means", chain, as.integer(step))
means <- means + (beta[,zeta+1] - means)/step
meansquares <- meansquares + ((beta[,zeta+1])^2 - meansquares)/step
probs <- lapply(1:3, function(hyp) probs[hyp,] + (apply(C[[hyp]] %*% beta[,zeta+1] > 0, 2, min) - probs[hyp,])/step)
probs <- do.call(rbind, probs)

test_that("probs equal",{
  expect_equal(outC[[1]], as.numeric(probs))
})

test_that("means equal",{
  expect_equal(outC[[2]], as.numeric(means))
})

test_that("meansquares equal",{
  expect_equal(outC[[3]], as.numeric(meansquares))
})

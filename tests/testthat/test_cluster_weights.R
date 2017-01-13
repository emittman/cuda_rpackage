context("Calculating cluster weights")

G <- as.integer(1000)
K <- as.integer(1000)
V <- as.integer(2)
N <- as.integer(3)

pi <- runif(K)
pi <- pi/sum(pi)

tau2 <- 1/rexp(K)

yty <- rnorm(G)

bxty <- rnorm(G*K)

bxxb <- rnorm(K)

Rout <- sapply(1:G*K, function(i){
  idk <- i %% K
  idg <- floor(i / K)
  log(pi[idk]) + 0.5 * V*N * log(tau2[idk]) + .05 * tau2[idk] * (yty[idg] - 2*bxty[i] + bxxb[idk])
})

Cout <- .Call("Rcluster_weights", bxty, pi, tau2, yty, bxxb, G, V, N, K)

test_that("Weights are correct", {
  expect_equal(Rout, Cout)
})
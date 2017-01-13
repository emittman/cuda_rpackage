context("Calculating cluster weights")

G <- as.integer(10)
K <- as.integer(5)
V <- as.integer(2)
N <- as.integer(3)

pi <- runif(K)
pi <- pi/sum(pi)

tau2 <- 1/rexp(K)

yty <- rnorm(G)

bxty <- rnorm(G*K)

bxxb <- rnorm(K)

idk <- rep(1:K, times=G)
idg <- rep(1:G, each=K)
Rout <- log(pi[idk]) + V*N* log(tau2[idk]) + tau2[idk] * (yty[idg] - 2*bxty[1:(G*K)] + bxxb[idk])


Cout <- .Call("Rcluster_weights", bxty, pi, tau2, yty, bxxb, G, V, N, K)

test_that("Weights are correct", {
  expect_equal(Rout, Cout)
})
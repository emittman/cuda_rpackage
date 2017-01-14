context("Calculating cluster weights")

G <- as.integer(10)
K <- as.integer(5)
V <- as.integer(2)
N <- as.integer(3)

pi <- rep(1, K)
# pi <- pi/sum(pi)

tau2 <- rep(1, K)#/rexp(K)

yty <- rep(0, G)

bxty <- rep(1, G*K)#rnorm(G*K)

bxxb <- rep(1.1, K) #rnorm(K)

idk <- rep(1:K, times=G)
idg <- rep(1:G, each=K)
Rout <- log(pi[idk]) + 0.5 * V * N * log(tau2[idk]) + -0.5 * tau2[idk] * (yty[idg] - 2*bxty[1:(G*K)] + bxxb[idk])


Cout <- .Call("Rcluster_weights", bxty, pi, tau2, yty, bxxb, G, V, N, K)

test_that("Weights are correct", {
  expect_equal(Rout, Cout)
})
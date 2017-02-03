context("Calculating cluster weights")

G <- as.integer(1000)
K <- as.integer(50)
N <- as.integer(4)

times <- 10

plyr::ldply(1:times, function(i){
  pi <- rbeta(K, 1, 1)
  pi <- pi/sum(pi)
  
  tau2 <- 1/rexp(K, 1)
  
  yty <- (rnorm(G))^2
  
  bxty <- rnorm(G*K)
  
  bxxb <- (rnorm(K))^2
  
  idk <- rep(1:K, times=G)
  idg <- rep(1:G, each=K)
  Rout <- log(pi[idk]) + 0.5 * N * log(tau2[idk]) + -0.5 * tau2[idk] * (yty[idg] - 2*bxty[1:(G*K)] + bxxb[idk])
  
  
  Cout <- .Call("Rcluster_weights", bxty, pi, tau2, yty, bxxb, G, N, K)
  
  test_that("Weights are correct", {
    expect_equal(Rout, Cout)
  })
})
context("Calculating cluster weights")

G <- as.integer(1000)
V <- as.integer(3)
K <- as.integer(50)
N <- as.integer(4)*V

times <- 3

# plyr::ldply(1:times, function(i){
#   pi <- rbeta(K, 1, 1)
#   pi <- pi/sum(pi)
#   
#   tau2 <- 1/rexp(K, 1)
#   
#   yty <- (rnorm(G))^2
#   
#   bxty <- rnorm(G*K)
#   
#   bxxb <- (rnorm(K))^2
#   
#   idk <- rep(1:K, times=G)
#   idg <- rep(1:G, each=K)
#   Rout <- log(pi[idk]) + 0.5 * N * log(tau2[idk]) + -0.5 * tau2[idk] * (yty[idg] - 2*bxty[1:(G*K)] + bxxb[idk])
#   
#   
#   Cout <- .Call("Rcluster_weights", bxty, pi, tau2, yty, bxxb, G, N, K)
#   
#   test_that("Weights are correct, given bxxb and bxty", {
#     expect_equal(Rout, Cout)
#   })
# })

for(i in 1:times){
  pi <- rbeta(K, 1, 1)
  pi <- pi/sum(pi)
  
  tau2 <- 1/rexp(K, 1)
  
  zeta <- sample(0:(K-1), G, replace=T)

  beta <- matrix(rnorm(K*V), V, K)
  
  group <- rep(1:V, each=N/V)
  X <- (cbind(rep(1, V), matrix(round(rnorm((V-1)*V), 0), V, V-1)))[group,]
  
  y <- t(matrix(rnorm(G*N, X %*% beta[,zeta]), N, G))
  
  data <- formatData(y, X, transform_y = identity)
  
  priors <- formatPriors(K=K, prior_mean = c(0,0), prior_sd = c(10,10), alpha = 1, a = 1, b = 1)
  
  chain <- formatChain(beta, pi, tau2, zeta)

  Rout <- sapply(1:G, function(g){
    sapply(1:K, function(k){
      log(pi[k]) + N * 0.5 * log(tau2[k]) + -0.5 * tau2[k] * (data$yty[g] - 2 * t(beta[,k]) %*% data$xty[,g] + t(beta[,k]) %*% matrix(data$xtx,V,V) %*% beta[,k])
    })
  })
  
  Cout <- .Call("Rcluster_weights", data, chain, priors);

  test_that("Weights are correct, given bxxb and bxty", {
    expect_equal(as.numeric(Rout), Cout)
  })  
}
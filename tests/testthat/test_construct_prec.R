context("Testing construct precision")

K <- 5
G <- 10
V <- 2
N <- 5

X <- matrix(rnorm(N*V), N, V)
y <- matrix(rnorm(N*G), G, N)
data <- formatData(y, X, transform_y = identity)

lambda2 <- rlnorm(V)
mu_0 <- rnorm(V)
priors <- formatPriors(K, mu_0, 1/sqrt(lambda2), 1, 1, 1)

tau2 <- rlnorm(K)
zeta <- sample(0:(K-1), G, replace=T)
beta <- rep(0, V*K)
pi <- rep(1/K, K)
chain <- formatChain(beta, pi, tau2, zeta)

xtx_rep <- rep(data$xtx, times=K)
dim(xtx_rep) <- c(V, V, K)

Mk <- sapply(0:(K-1), function(k) sum(zeta == k))

xtx_Mk <- xtx_rep * rep(Mk, each=V*V)

data$xtx <- xtx_Mk

Rprec <- sapply(1:K, function(k){
  submat <- xtx_Mk[,,k] * tau2[k]
  if(V == 1){
    submat <- submat + lambda2
  } else{
    diag(submat) <- diag(submat) + lambda2
  }
  submat
})
  
dim(Rprec) <- c(V,V,K)
  
  
Cprec <- Rconstruct_prec(data, priors, chain, as.integer(1))
  
test_that("Correct values",{
  expect_equal(Rprec, Cprec)
})
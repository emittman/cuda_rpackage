context("Testing construct precision")

K <- 100
G <- 200
V <- 2
N <- 5

X <- matrix(rnorm(N*V), N, V)
y <- matrix(rnorm(N*G), G, N)

lambda2 <- rlnorm(V)
mu_0 <- rnorm(V)
priors <- formatPriors(K, mu_0, 1/sqrt(lambda2), 1, 1, 1)

tau2 <- rlnorm(K)
zeta <- sample(0:(K-1), G, replace=T)
beta <- rep(0, V*K)
pi <- rep(1/K, K)
alpha <- 1

chain <- formatChain(beta, pi, tau2, zeta, alpha)
data1 <- formatData(y, X, transform_y = identity, test_voom=T)
data2 <- formatData(y, X, transform_y = identity, test_voom=F)

xtx_rep <- rep(data2$xtx, times=K)
dim(xtx_rep) <- c(V, V, K)

Mk <- sapply(0:(K-1), function(k) sum(zeta == k))

xtx_Mk <- xtx_rep * rep(Mk, each=V*V)


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
  
Cprec1 <- Rconstruct_prec(data1, priors, chain)
Cprec2 <- Rconstruct_prec(data2, priors, chain)

test_that("Correct values",{
  expect_equal(Rprec, Cprec1)
  expect_equal(Rprec, Cprec2)
})
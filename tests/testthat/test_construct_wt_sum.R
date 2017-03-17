context("Construct weighted sum")
G <- as.integer(10)
K <- as.integer(5)
V <- as.integer(2)
N <- as.integer(10)

y <- matrix(rnorm(G*N), G, N)
X <- matrix(rnorm(N*V), N, V)
data <- formatData(y, X, transform_y=identity)

beta <- matrix(rnorm(K*V),V,K)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rexp(K)
zeta <- as.integer(sample(0:(K-1), G, replace=T))
chain <- formatChain(beta, pi, tau2, zeta)

priors <- formatPriors(K, rcauchy(V), rexp(V), 1, 1, 1)

Csum <- .Call("Rtest_weighted_sum", data, priors, chain, verbose=as.integer(0))
dim(Csum) <- c(V, K)

xty_sums <- sapply(1:K, function(k){
  submat = data$xty[,which(zeta+1 == k)]
  if(length(submat)>V) return(rowSums(submat))
  if(length(submat)>0) return(submat)
  else return(rep(0, V))
})

Rsum <- xty_sums * rep(chain$tau2, each=V) + rep(priors$mu_0, K) * rep(priors$lambda2, K)

test_that("Correct answer", {
  expect_equal(Csum, Rsum)
})


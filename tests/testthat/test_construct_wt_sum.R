context("Construct weighted sum")
G <- as.integer(10)
K <- as.integer(500)
V <- as.integer(5)
N <- as.integer(10)

data <- formatData(y=matrix(rnorm(G*N), G, N),
                   X = matrix(rnorm(N*V), N, V), tranform_y=identity)

beta <- matrix(rnorm(K*V),V,K)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rexp(K)
zeta <- as.integer(sample(0:(K-1), G, replace=T))
chain <- formatChain(beta, pi, tau2, zeta)

priors <- formatPriors(K, rcauchy(V), rexp(V), 1, 1, 1)

Cmean <- .Call("Rtest_wt_sum", priors, chain)
Rmean <- rep(priors$mu_0, K) * rep(priors$lambda2, K) / rep(chain$tau2,each=V)

test_that("Correct answer", {
  expect_equal(Cmean, Rmean)
})


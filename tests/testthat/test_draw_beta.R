context("Beta full conditional")
G <- as.integer(300)
V <- as.integer(1)
N <- as.integer(20)
n_iter <- as.integer(100)

seed <- as.integer(sample(1e5, 1))
K <- G

zeta <- as.integer(0:(G-1))

X <- matrix(rep(1, N),N,V)

beta <- seq(1, 100, length.out = K)

y <- t(matrix(rnorm(G*N, X %*% beta[zeta+1]), N, G))

data <- formatData(y, X, transform_y = identity)
priors <- formatPriors(K=K, prior_mean = 0, prior_sd = 100, a = 1, b = 1)

pi_init <- rep(1/K, K)
zeta_init <- zeta
tau2_init <- rep(1, K)
alpha <- 1

chain_init <- formatChain(beta, pi_init, tau2_init, zeta, alpha)
idx_save <- as.integer(0:(G-1))
Cout <- .Call("Rtest_draw_beta", chain_init, data, priors, n_iter, idx_save, seed)
expected <- data.frame(beta = data$xty / (N + priors$lambda))
samples <- NULL
samples[["beta"]] <- array(Cout[[1]], dim = c(V, G, n_iter))
estimate <- apply(samples$beta, c(1, 2), mean)
se <- sqrt(1/(n_iter*(N+priors$lambda)))

test_that("Averages match expectations", {
  test_stat <- sum(((estimate - expected)/se)^2)
  expect_lt(test_stat, qchisq(.99, G))
  expect_gt(test_stat, qchisq(.01, G))
})


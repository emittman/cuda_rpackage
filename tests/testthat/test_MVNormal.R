context("Conditiona MVN sampling")
set.seed(102030)
seed <- as.integer(10101)
K <- 4
extraK <- 2
reps <- 1
V <- 2
n_per_v <- 20
group <- rep(1:K, each=reps)
zeta <- as.integer(rep(0:(K-1), each=reps))

X <- kronecker(diag(V), rep(1, n_per_v))

beta <- matrix(c(1, 1,
                 1, -1,
                 -1, -1,
                 -1, 1), V, K)

y <- t(matrix(rnorm(K*V*n_per_v*reps, X %*% beta[,group]), V*n_per_v, K*reps))

data <- formatData(y, X, transform_y = identity)

priors <- formatPriors(K=K+extraK, prior_mean = c(0,0), prior_sd = c(10,10), alpha = 1, a = 1, b = 1)

out <- .Call("Rtest_MVNormal", seed, zeta, data, priors)

print("true beta")
print(beta)
print("beta_draws:")
print(matrix(out, 2, 4))
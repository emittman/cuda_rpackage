context("Conditiona MVN sampling")

seed <- as.integer(10101)
K <- 4
reps <- 10
V <- 2
n_per_v <- 3
group <- rep(1:K, each=reps)
zeta <- as.integer(rep(1:4, each=reps))

X <- kronecker(diag(V), rep(1, n_per_v))

beta <- matrix(c(1, 1,
                 1, -1,
                 -1, -1,
                 -1, 1), 2, 4)

y <- t(matrix(rnorm(K*V*n_per_v*reps, X %*% beta[,group]), V*n_per_v, K*reps))

data <- formatData(y, X, transform_y = identity)

priors <- formatPriors(K=K, prior_mean = c(0,0), prior_sd = c(10,10), alpha = 1, a = 1, b = 1)

out <- .Call("Rtest_MVNormal", seed, zeta, data, priors)

print("true beta")
print(beta)
print("beta_draws:")
print(matrix(out, 2, 4))
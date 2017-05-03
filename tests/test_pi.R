set.seed(13117)

K <- as.integer(500)
G <- as.integer(1000)
V <- as.integer(2)

seed <- as.integer(sample(1e4, 1))

n_per_v <- 5

zeta <- rep(0:499, each=2)
alpha <- 10

#filler
X <- kronecker(diag(V), rep(1, n_per_v))
beta <- matrix(seq(1, 5, length.out = V*K), V, K)
y <- t(matrix(rnorm(G*V*n_per_v, X %*% beta[,(zeta+1)], .1), V*n_per_v, G))

data <- formatData(y, X, transform_y = identity)

chain <- formatChain(beta, rep(1/K, K), rep(1.2, K), as.integer(zeta))

priors <- formatPriors(K, c(0,0), c(1,1), alpha, 1, 1)

#print(zeta)
print(table(zeta))
.Call("Rtest_draw_pi", seed, chain, priors, data, as.integer(1))
#extern "C" SEXP Rtest_draw_beta(SEXP Rseed, SEXP Rchain, SEXP Rpriors, SEXP Rdata){
  
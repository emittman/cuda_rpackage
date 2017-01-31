set.seed(13117)

K <- as.integer(4)
G <- as.integer(10)
V <- as.integer(2)

seed <- as.integer(sample(1e4, 1))

n_per_v <- 5

zeta <- sample(0:(K-1), G, replace=T)

X <- kronecker(diag(V), rep(1, n_per_v))

beta <- matrix(seq(1, 5, length.out = V*K), V, K)

y <- t(matrix(rnorm(G*V*n_per_v, X %*% beta[,group], .1), V*n_per_v, G))

data <- formatData(y, X, transform_y = identity)

chain <- formatChain(beta, rep(1/K, K), rep(1.2, K), as.integer(zeta))

priors <- formatPriors(K, c(0,0), c(1,1), 1, 0.1, .3)

print("zeta:\n")
print(zeta)
print("Mk:\n")
table(zeta)
print("Tk:\n")
.Call("Rtest_draw_pi", seed, chain, priors, data)
#extern "C" SEXP Rtest_draw_beta(SEXP Rseed, SEXP Rchain, SEXP Rpriors, SEXP Rdata){
  
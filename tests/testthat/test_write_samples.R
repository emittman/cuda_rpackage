context("Write samples")

G <- as.integer(10)
K <- as.integer(5)
V <- as.integer(2)

n_iter <- as.integer(4)

beta <- matrix(rnorm(K*V),V,K)
pi <- rbeta(K, 1, 9)
pi <- pi/sum(pi)
tau2 <- rgamma(K, 2, 2)
zeta <- as.integer(sample(0:(K-1), G, replace=T))

G_save <- 5
idx <- as.integer(sample(0:(G-1), G_save))

chain <- formatChain(beta, pi, tau2, zeta)

Cout <- .Call("Rtest_write_samples", chain, idx, n_iter)

  
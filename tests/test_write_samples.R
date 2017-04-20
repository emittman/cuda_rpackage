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

Rout <- list(beta = rep(beta[,(zeta[idx+1]+1)], n_iter),
             tau2 = rep(tau2[zeta[idx+1]+1], n_iter),
             P = rep(c(pi, as.numeric(beta)), n_iter),
             max_id = rep(max(zeta), n_iter),
             num_occ = rep(length(unique(zeta)), n_iter)
            )

test_that("betas match", {expect_equal(as.numeric(Rout$beta), Cout[[1]])})
test_that("tau2s match", {expect_equal(as.numeric(Rout$tau2), Cout[[2]])})
test_that("P matches", {expect_equal(as.numeric(Rout$P), Cout[[3]])})
test_that("max_id matches", {expect_equal(Rout$max_id, Cout[[4]])})
test_that("num_occ matches", {expect_equal(Rout$num_occ, Cout[[5]])})
context("Solve many normal equations in parallel")

K <- as.integer(1000)
V <- as.integer(4)

Xs <- lapply(1:K, function(k) matrix(rnorm(4*V*V), 4*V, V))
ys <- lapply(1:K, function(k) rnorm(4*V))

xtys <- lapply(1:K, function(k) t(Xs[[k]]) %*% ys[[k]])

Ls <- lapply(1:K, function(k) t(chol(t(Xs[[k]]) %*% Xs[[k]])))

Rsols <- unlist(lapply(1:K, function(k) solve(t(Xs[[k]]) %*% Xs[[k]]) %*% xtys[[k]]))
Csols <- .Call("Rbeta_hat", unlist(Ls), unlist(xtys), K, V)

test_that("Solved normal equations", {
  expect_equal(Rsols, Csols)
})
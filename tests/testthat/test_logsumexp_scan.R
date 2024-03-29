context("Weight normalization with log_sum_exp")

dim <- 1e2

x <- matrix(rnorm(dim^2), dim, dim)

C_res <- matrix(.Call("Rnormalize_wts", x, as.integer(dim), as.integer(dim)), dim, dim)
R_res <- log(apply(exp(x), 2, cumsum))

test_that("Result is correct", {
  expect_equal(C_res, R_res)
})
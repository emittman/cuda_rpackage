context("Test rgamma, rbeta functions")

n <- 1e3

n_reps <- 100

values <- 10

for(i in 1:values) {
  
  a <- ifelse(i %% 2 == 0, runif(1), abs(rcauchy(1)) + 1)
  b <- abs(rcauchy(1))
  
  mu <- a/b
  mu_se <- sqrt(a/b^2/n)
  gamma_c <- lapply(1:n_reps, function(rep) rgamma(n, a, b))
  
  # gamma_c <- lapply(1:n_reps, function(rep) .Call("Rgamma_rng", rep(a, n), rep(b, n)))
  in_range <- sum(sapply(gamma_c, function(rep) abs((mean(rep)-mu)/mu_se) < 2))
  
  test <- c(in_range > qbinom(.005, n_reps, .95), in_range < qbinom(.995, n_reps, .95))

  if(!(all(test))){
    chance <- 1 - (.99)^values
    sprintf("Stochastic test, can fail due to random chance (%.2f chance)", round(chance,2))
  }
  
  test_that("Results match expectations", {
    expect_equal(test, c(TRUE, TRUE))
  })
}  
  
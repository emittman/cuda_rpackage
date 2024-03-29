context("Running means")
set.seed(2817)
pows <- as.integer(c(1,2,3))

len <- as.integer(100)
N <- as.integer(30)

X <- matrix(rnorm(len*N), len, N);

for(p in pows){

  Rout <- drop(apply(X, 1, function(row) mean(row^p)))
  Cout <- X[,1]^p
  for(i in 2:N){
    Cout <- .Call("Rtest_running_mean", Cout, X[,i], p, as.integer(i))
  }
  
  test_that(paste(c("Means match. (power = ", p, ")"), collapse=""), {
    expect_equal(Rout, Cout)
  })
  
}
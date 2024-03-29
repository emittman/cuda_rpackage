context("Testing cholesky decomposition")

dim <- 5
n <- 100

#create random symmetric positive definite matrix
make_spd <- function(dim){
  x <- matrix(round(rnorm(dim^2)*3, 2), dim, dim)
  return(x %*% t(x))
}

# Create an array of spd matrices
make_spd_array <- function(dim, n){
  y <- sapply(1:n, function(x) as.numeric(make_spd(dim)))
  dim(y) <- c(dim, dim, n)
  return(y)
}

# Find lower cholesky facotorization of all spd matrices in array
Rchol <- function(spd_array){
  d <- dim(spd_array)
  y <- apply(spd_array, 3, function(x) t(chol(matrix(x, d[1], d[2]))))
  dim(y) <- d
  return(y)
}

# Are lower triangular parts of two arrays are equal?
compare_chol <- function(sub_array1, sub_array2, dim){
  tri <- lower.tri(sub_array1)
  return(all.equal(tri*sub_array1, tri*sub_array2))
}


# Compare (lower triangular part of) all 2 dim submatrices in array of matrices
do_compare <- function(spd_array, n){
  chol1 <- Rchol(spd_array)
  chol2 <- Rchol_multiple(spd_array)
  result <- sapply(1:n, function(x) compare_chol(chol1[,,x], chol2[,,x]))
  return(result)
}

test_that("factorization is correct", {
  A <- make_spd_array(dim, n)
  bool <- do_compare(A, n)
  expect_equal(all(bool), TRUE)
})

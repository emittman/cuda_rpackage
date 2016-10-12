make_spd <- function(dim){
  x <- matrix(round(rnorm(dim^2)*5, 2), dim, dim)
  return(x %*% x)
}

make_spd_array <- function(dim, n){
  sapply(
}
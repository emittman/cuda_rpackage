#' @title Function \code{ratio.cube.to.sphere}
#' @param n number of dimensions

ratio.cube.to.sphere <- function(n){
  n <- as.integer(n)
  stopifnot(n>0)
  n * 2^(n-1) * gamma(n/2) / pi^(n/2)
}

#' @title Function \code{initFixed}
#' @description Produce a formatted list for control arguments
#' @param priors
#' @param estimates
#' @param C
#' @export

initFixedGrid <- function(priors, estimates, C=NULL){
  G <- length(estimates$sigma2)
  p <- length(priors$mu_0)+1
  #combine betas and log(sigma)
  param <- with(estimates, rbind(beta, .5*log(sigma2)))
  #get range of empirical estimates
  ranges <- apply(param, 1, range)
  #Include extra points to be trimmed so that what remains resembles
  #'ball' of uniformly spaced points
  K_corr <- ratio.cube.to.sphere(p) * priors$K
  grid_size <- ceiling(K_corr^(1/p))
  grid <- data.frame(apply(ranges, 2, function(r){
    seq(r[1], r[2], length.out=grid_size)
  }))
  grid.full <- expand.grid(grid)
  ###arrange by prior density
  # log(sigma) -> 1/sigma^2
  grid.full[,p] <- exp(-2*grid.full[,p])
  grid.full$dens <- mvtnorm::dmvnorm(data.matrix(grid.full[,-p]), priors$mu_0, diag(1/priors$lambda2))*
    dgamma(grid.full[,p], priors$a, priors$b)
  param <- t(data.matrix(dplyr::arrange(grid.full, desc(dens))[1:priors$K,-(p+1)]))
  beta <- param[1:(p-1),]
  tau2 <- param[p,]
  pi <- with(priors, rep(1/K, K))
  zeta <- with(priors, as.integer(sample(K, G, replace=T) - 1)) #overwritten immediately
  alpha <- with(priors, rgamma(1, A, B))
  formatChain(beta, pi, tau2, zeta, alpha, C)
}
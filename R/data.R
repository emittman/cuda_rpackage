#' @title Function \code{formatData}
#' @description Reformat the data
#' @export
#' @param counts matrix of counts
#' @param X number of vectors
#' @param groups identifies columns with treatments

formatData <- function(counts, X, groups = NULL, transform_y = function(x) log(x + 1)){
  adjustX = FALSE
  if(nrow(X) != ncol(counts)){
    if(is.null(groups))  stop("groups must be specified!")
    if(length(groups) != ncol(counts)) stop("length(groups) must equal ncol(counts)!")
    adjustX = TRUE
  }
  
  if(adjustX){
    X <- X[groups,]
  }
  
  G <- nrow(counts)
  V <- ncol(X)
  N <- nrow(X)

  y <- transform_y(counts)  
  xty <- apply(y, 1, function(y) t(y) %*% X)
  yty <- drop(apply(y, 1, crossprod))
  xtx <- as.numeric(t(X) %*% X)
  
  data = list(yty = yty, xty = xty, xtx = xtx, G = as.integer(G), V = as.integer(V), N = as.integer(N))
  return(data)
}

#' @title Function \code{formatPriors}
#' @description format priors
#' @export
#' @param K number stick-breaking components
#' @param prior_mean numeric
#' @param prior_sd numeric matching dimension of prior_mean
#' @param alpha mass parameter
#' @param a prior shape for error precision
#' @param b prior scale for error precision

formatPriors <- function(K, prior_mean, prior_sd, alpha, a, b){
  K <- as.integer(K)
  if(K < 1) stop("K must be postive!")
  if(length(prior_mean) != length(prior_sd)){
    stop("Check dimensions of prior!")
  }
  if(alpha<=0) stop("alpha must be positive!")
  if(a<=0) stop("a must be postive!")
  if(b<=0) stop("b must be postive!")
  list(K, V = as.integer(length(prior_mean)),
       as.numeric(prior_mean),
       1/as.numeric(prior_sd)^2,
       alpha, a, b)
}

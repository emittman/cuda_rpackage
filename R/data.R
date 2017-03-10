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
  list(K = K, V = as.integer(length(prior_mean)),
       mu_0 = as.numeric(prior_mean),
       lambda2 = 1/as.numeric(prior_sd)^2,
       alpha = as.numeric(alpha), a = as.numeric(a), b = as.numeric(b))
}

#' @title Function \code{formatChain}
#' @description format chain state
#' @export
#' @param beta V*K array
#' @param pi K array in $(0, 1)$
#' @param tau2 K array in $(0, ...)$
#' @param zeta G array in $\{0,...,K-1\}$
#' @param C list of matrices with ncol = V
#' @param probs G array of probabilities
#' @param means G*V array
#' @param meansquares G*V array

formatChain <- function(beta, pi, tau2, zeta, C=NULL, probs=NULL, means=NULL, meansquares=NULL){
  G = as.integer(length(zeta))
  V = as.integer(length(beta)/length(pi))
  K = as.integer(length(beta)/V)
  
  if(max(zeta)>K-1) stop("C uses zero indexing")
  
  if(!is.null(C)){
    if(!is.list(C)){
      C <- list(C)
    }
    if(!all(sapply(C, is.matrix)))                stop("C must be a matrix or a list of matrices")
    if(!all(sapply(C, function(h) ncol(h) == V))) stop("All elements of C must have same number of columns")
    n_hyp <- length(C)
    C_rowid <- as.integer(rep(1:n_hyp, sapply(C, nrow)) - 1)
    Cmat <- do.call(rbind, C)
    P <- nrow(Cmat)
  } else {
    n_hyp <- as.integer(1)
    C_rowid <- as.integer(rep(0, V))
    Cmat <- diag(V)
    P <- V
  }
  if(!is.null(probs)){
    if(length(probs) != n_hyp*G) stop("probs doesn't match C and/or G")
  } else {
    probs = rep(0, n_hyp*G)
  }
  if(!is.null(means)){
    if(length(means) != V*G) stop("means is wrong dimension")
  } else{
    means = rep(0, V*G)
  }
  if(!is.null(meansquares)){
    if(length(meansquares) != V*G) stop("means is wrong dimension!")
  } else{
    meansquares = rep(0, V*G)
  }
  list(G = G, V = V, K = K, n_hyp = n_hyp, C_rowid, P = P, beta = as.numeric(beta), pi = as.numeric(pi), tau2 = as.numeric(tau2),
       zeta = as.integer(zeta), C = t(Cmat), probs = probs, means = means, meansquares = meansquares)
}


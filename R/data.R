#' @title Function \code{formatData}
#' @description Reformat the data
#' @export
#' @param counts matrix of counts
#' @param X number of vectors
#' @param groups identifies columns with treatments

formatData <- function(counts, X, groups = NULL, transform_y = function(x) log(x + 1), voom=FALSE){
  adjustX = FALSE
  if(nrow(X) != ncol(counts)){
    stopifnot(!is.null(groups), length(groups) == ncol(counts))
    adjustX = TRUE
  }
  
  if(adjustX){
    X <- X[groups,]
  }
  
  G <- nrow(counts)
  V <- ncol(X)
  N <- nrow(X)
  if (voom & requireNamespace("limma", quietly = TRUE)) {
    voom_out <- limma::voomWithQualityWeights(counts, design=X,
                                            nomalization="none",
                                            plot = FALSE)
    y <- voom_out[[1]]
    W <- voom_out[[2]]
    ytWy <- drop(sapply(1:G, function(g) y[g,] %*% diag(W[g,]) %*% y[g,]))
    xtWy <- sapply(1:G, function(g) y[g,] %*% diag(W[g,]) %*% X)
    xtWx <- sapply(1:G, function(g) t(X) %*% diag(W[g,]) %*% X)
    data = list(yty = ytWy, xty = xtWy, xtx = xtWx, G = as.integer(G),
                V = as.integer(V), N = as.integer(N), voom=voom)
  } else{
    if(voom) print("limma is not installed, defaulting to unweighted version")
      y <- transform_y(counts)  
      xty <- apply(y, 1, function(y) t(y) %*% X)
      yty <- drop(apply(y, 1, crossprod))
      xtx <- as.numeric(t(X) %*% X)
      data = list(yty = yty, xty = xty, xtx = xtx, G = as.integer(G),
                  V = as.integer(V), N = as.integer(N), voom=voom)
  } 
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

formatPriors <- function(K, prior_mean, prior_sd, alpha=NULL, a=1, b=1, A=1, B=1){
  stopifnot(K >= 1, length(prior_mean)==length(prior_sd), a>0, b>0, A>0, B>0)
  if(is.null(alpha)){
    alpha = rgamma(1, A, B)
  } else {
    stopifnot(alpha>0)
  }
  list(K       = as.integer(K),
       V       = as.integer(length(prior_mean)),
       mu_0    = as.numeric(prior_mean),
       lambda2 = 1/as.numeric(prior_sd)^2,
       a       = as.numeric(a),
       b       = as.numeric(b),
       alpha   = as.numeric(alpha),
       A       = as.numeric(A),
       B       = as.numeric(B))
}

#' @title Function \code{formatChain}
#' @description Produce a list to be used as 'chain' argument for mcmc()
#' @export
#' @param beta numeric matrix with V rows and K columns; columns are cluster locations
#' @param pi numeric vector, length K, taking values in $(0, 1)$
#' @param tau2 numeric vector, length K, taking positive values
#' @param zeta integer vector, length G, with values in the range $\{0,...,K-1\}$
#' @param C list of matrices with V columns. Each matrix represents a hypothesis which is
#'   true, for a given cluster, iff all C_i * beta_g > 0 == TRUE
#' @param probs numeric vector, length G*length(C), representing probabilities of the hypotheses
#'   encoded in C for all G genes in column-major order
#' @param means numeric vector, length G*V, representing latent posterior mean for location parameter
#'  for all G genes in column-major order
#' @param meansquares numeric vector, length G*V 
#' @param s_RW_alpha numeric, standard deviation for random walk Metropolis when
#'   \code{!alpha_fixed} and weightsMethod = "symmDirichlet". Defaults to 0.

formatChain <- function(beta, pi, tau2, zeta, C=NULL, probs=NULL, means=NULL, meansquares=NULL, s_RW_alpha=0){
  G = as.integer(length(zeta))
  V = as.integer(length(beta)/length(pi))
  K = as.integer(length(beta)/V)
  
  stopifnot(max(zeta) < K, min(zeta) < 0)
  
  if(!is.null(C)){
    if(!is.list(C)){
      C <- list(C)
    }
    stopifnot(all(sapply(C, is.matrix)), all(sapply(C, function(h) ncol(h) == V)))
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
    stopifnot(length(probs) == n_hyp*G)
  } else {
    probs = rep(0, n_hyp*G)
  }
  if(!is.null(means)){
    stopifnot(length(means) == V*G)
  } else{
    means = rep(0, V*G)
  }
  if(!is.null(meansquares)){
    stopifnot(length(meansquares) == V*G)
  } else{
    meansquares = rep(0, V*G)
  }
  list(G = G, V = V, K = K, n_hyp = n_hyp, C_rowid = C_rowid, P = P, beta = as.numeric(beta), pi = as.numeric(log(pi)), tau2 = as.numeric(tau2),
       zeta = as.integer(zeta), C = t(Cmat), probs = probs, means = means, meansquares = meansquares, s_RW_alpha=s_RW_alpha)
}

#'@title Function \code{initChain}
#' @description initialize chain using prior
#' @export
#' @param priors list, from formatPriors
#' @param G integer
#' @param C list

initChain <- function(priors, G, C=NULL){
  beta <- with(priors, matrix(rnorm(V*K, rep(mu_0, K), rep(1/sqrt(lambda2), K)), V, K))
  tau2 <- with(priors, rgamma(K, a, b))
  pi <- with(priors, rexp(K))
  pi <- pi/sum(pi)
  zeta <- with(priors, as.integer(sample(K, G, replace=T) - 1))
  formatChain(beta, pi, tau2, zeta, C)
}

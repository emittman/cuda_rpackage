#' @title Function \code{formatData}
#' @description Reformat the data
#' @export
#' @param counts matrix of counts
#' @param X number of vectors
#' @param groups identifies columns with treatments
#' @param transform_y function specifying how to transform the counts. Defaults to log(x+1). If voom==TRUE,
#' then transform_y is ignored
#' @param voom Logical value indicating whether to compute Voom precision weights
#' @param test_voom logical Only used for testing purposes. If true, forced separate xtx matrices in memory

formatData <- function(counts, X, groups = NULL, transform_y = function(x) log(x + 1), voom=FALSE, test_voom=FALSE){
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
  if(test_voom) cat("Testing voom by simply setting all W_g to 1.\n")
  if (voom & requireNamespace("limma", quietly = TRUE) & !test_voom) {
    cat("Computing precision weights with voom ...")
    voom_out <- limma::voomWithQualityWeights(counts, design=X,
                                            nomalization="none",
                                            plot = FALSE)
    cat("done.\n")
    y <- voom_out[[1]]
    W <- voom_out[[2]]
    ytWy <- drop(sapply(1:G, function(g) y[g,] %*% diag(W[g,]) %*% y[g,]))
    xtWy <- sapply(1:G, function(g) y[g,] %*% diag(W[g,]) %*% X)
    xtWx <- sapply(1:G, function(g) t(X) %*% diag(W[g,]) %*% X)
    data = list(yty = ytWy, xty = xtWy, xtx = xtWx, G = as.integer(G),
                V = as.integer(V), N = as.integer(N), voom=voom,
                transformed_counts = y, X = X)
  } else {
    if(voom) print("limma is not installed, defaulting to unweighted version")
      y <- transform_y(counts)  
      xty <- apply(y, 1, function(y) t(y) %*% X)
      yty <- drop(apply(y, 1, crossprod))
      xtx <- as.numeric(t(X) %*% X)
      if(test_voom){
        xtx <- rep(xtx, times=G)
      }
      data = list(yty = yty, xty = xty, xtx = xtx, G = as.integer(G),
                  V = as.integer(V), N = as.integer(N), voom=as.logical(voom+test_voom),
                  transformed_counts = y, X = X)
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

formatPriors <- function(K, prior_mean=NULL, prior_sd=NULL, a=1, b=1, A=1, B=1, estimates=NULL){
  # Checking input
  stopifnot(K >= 1, length(prior_mean)==length(prior_sd), a>0, b>0, A>0, B>0)
  if(is.null(prior_mean) & is.null(prior_sd) & is.null(estimates)) stop("Need to specify more parameters")
  
  if(!is.null(estimates)){
    print("Estimating priors...")
    estPriors <- informPriors(estimates)
    print("done.")
  } else{
    estPriors <- NULL
  }

  with(estPriors, list(
    K       = as.integer(K),
    V       = as.integer(length(prior_mean)),
    mu_0    = as.numeric(prior_mean),
    lambda2 = 1/as.numeric(prior_sd)^2,
    a       = as.numeric(a),
    b       = as.numeric(b),
    A       = as.numeric(A),
    B       = as.numeric(B))
  )
}

#' @title Function \code{formatChain}
#' @description Produce a list to be used as 'chain' argument for mcmc()
#' @export
#' @param beta numeric matrix with V rows and K columns; columns are cluster locations
#' @param pi numeric vector, length K, taking values in $(0, 1)$
#' @param tau2 numeric vector, length K, taking positive values
#' @param zeta integer vector, length G, with values in the range $\{0,...,K-1\}$
#' @param alpha numeric, positive
#' @param C list of matrices with V columns. Each matrix represents a hypothesis which is
#'   true, for a given cluster, iff all C_i * beta_g > 0 == TRUE
#' @param probs numeric vector, length G*length(C), representing probabilities of the hypotheses
#'   encoded in C for all G genes in column-major order
#' @param means numeric vector, length G*V, representing latent posterior mean for location parameter
#'  for all G genes in column-major order
#' @param meansquares numeric vector, length G*V 
#' @param s_RW_alpha numeric, standard deviation for random walk Metropolis when
#'   \code{!alpha_fixed} and weightsMethod = "symmDirichlet". Defaults to 0.

formatChain <- function(beta, pi, tau2, zeta, alpha, C=NULL, probs=NULL, means=NULL, meansquares=NULL,
                        slice_width=1, max_steps=100){
  G = as.integer(length(zeta))
  V = as.integer(length(beta)/length(pi))
  K = as.integer(length(beta)/V)
  
  stopifnot(max(zeta) < K, min(zeta) >= 0, alpha > 0, min(tau2) > 0, min(pi) > 0, max(pi) < 1)
  
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
       zeta = as.integer(zeta), alpha = as.numeric(alpha), C = t(Cmat), probs = probs, means = means, meansquares = meansquares, 
       slice_width = as.numeric(slice_width), max_steps = as.integer(max_steps))
}

#'@title Function \code{initChain}
#' @description initialize chain using prior
#' @export
#' @param priors list, from formatPriors
#' @param G integer
#' @param C list

initChain <- function(priors, G, C=NULL, estimates=NULL){
  if(length(estimates)){
    init_id <- sort(sample(G, priors$K))
    beta <- estimates[[1]][,init_id]
    tau2 <- 1/estimates[[2]][init_id]
  } else{
    beta <- with(priors, matrix(rnorm(V*K, rep(mu_0, K), rep(1/sqrt(lambda2), K)), V, K))
    tau2 <- with(priors, rgamma(K, a, b))    
  }
  pi <- with(priors, rep(1/K, K))
  zeta <- with(priors, as.integer(sample(K, G, replace=T) - 1)) #overwritten immediately
  alpha <- with(priors, rgamma(1, A, B))
  formatChain(beta, pi, tau2, zeta, C)
}

#' @title Function \code{indEstimates}
#' @description compute independent estimates of gene specific parameters. Used for 
#' computing informative priors and initializing chain
#' @param data list, from formatData

indEstimates <- function(data){
  betas <- with(data, sapply(1:G, function(g){
    qr.solve(qr(matrix(xtx[1:(V*V) + (g-1)*voom*(V*V)],V,V)), xty[,g])
  }))
  sigma2s <- with(data, sapply(1:G, function(g){
    (yty[g] - 2*t(xty[,g]) %*% betas[,g] + 
       t(betas[,g]) %*% matrix(xtx[1:(V*V) + (g-1)*voom*(V*V)],V,V) %*% betas[,g])/(N-V)
  }))
  return(list(beta=betas, sigma2=sigma2s))
}

#' @title Function \code{informPriors}
#' @description select priors to put mass over the range of the data
#' @param estimates list, from indEstimates
#' @return a named list to be passed to formatPriors
#' 
informPriors <- function(estimates){
  V <- dim(estimates[[1]])[1]
  pr_tau2_var <- var(1/estimates[[2]] + .01)
  pr_tau2_mean <- mean(1/estimates[[2]] + .01)
  list(prior_mean = sapply(1:V, function(v) median(estimates[[1]][v,])),
       prior_sd = sapply(1:V, function(v) 2 * sd(estimates[[1]][v,])),
       a = pr_tau2_mean^2 / pr_tau2_var,
       b = pr_tau2_mean   / pr_tau2_var)
}


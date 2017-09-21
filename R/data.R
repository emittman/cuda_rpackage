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

formatData <- function(counts, X, groups = NULL, transform_y = function(x) log(x + 1), voom=FALSE, normalize=FALSE, test_voom=FALSE){
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
  if(normalize & !voom) cat("TMM normalization without voom is not supported by this function.\nNormalization should be performed manually prior to calling formatData().")
  if(test_voom) cat("Testing voom by simply setting all W_g to 1.\n")
  if (voom & requireNamespace("limma", quietly = TRUE) & !test_voom) {
    if(normalize & requireNamespace("edgeR", quietly = TRUE)){
      cat("Computing scale normalization factors (by sample) using TMM method.")
      counts <- edgeR::DGEList(counts)
      counts <- edgeR::calcNormFactors(counts)
      cat("Computing precision weights with voom ...")
      voom_out <- limma::voom(counts, design=X, normalize.method="none", plot = FALSE)
      cat("done.\n")
      y <- voom_out[[2]]
      W <- voom_out[[3]]
    } else{
      cat("Computing precision weights with voom ...")
      voom_out <- limma::voom(counts, design=X, normalize.method="none", plot = FALSE)
      cat("done.\n")
      y <- voom_out[[1]]
      W <- voom_out[[2]] 
    }
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
                  V = as.integer(V), N = as.integer(N), voom=as.logical(test_voom),
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
#' @param a prior shape for error precision
#' @param b prior scale for error precision
#' @param A prior shape for mass parameter
#' @param B prior scale for mass parameter
#' @param estimates optional, if provided, data is used to inform prior

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
#' @param means_betas numeric vector, length G*V, representing latent posterior mean for location parameter
#'  for all G genes in column-major order
#' @param meansquares_betas numeric vector, length G*V 
#' @param means_sigmas numeric vector, length G, representing latent posterior mean for scale parameter
#'  for all G genes in column-major order
#' @param meansquares_sigmas numeric vector, length G 
#' @param s_RW_alpha numeric, standard deviation for random walk Metropolis when
#'   \code{!alpha_fixed} and weightsMethod = "symmDirichlet". Defaults to 0.

formatChain <- function(beta, pi, tau2, zeta, alpha, C=NULL, probs=NULL, means_betas=NULL, meansquares_betas=NULL, means_sigmas=NULL,
                        meansquares_sigmas=NULL,slice_width=1, max_steps=100){
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
  if(!is.null(means_betas)){
    stopifnot(length(means_betas) == V*G)
  } else{
    means_betas = rep(0, V*G)
  }
  if(!is.null(meansquares_betas)){
    stopifnot(length(meansquares_betas) == V*G)
  } else{
    meansquares_betas = rep(0, V*G)
  }
  if(!is.null(means_sigmas)){
    stopifnot(length(means_sigmas) == G)
  } else{
    means_sigmas = rep(0, G)
  }
  if(!is.null(meansquares_sigmas)){
    stopifnot(length(meansquares_sigmas) == G)
  } else{
    meansquares_sigmas = rep(0, G)
  }
  list(G = G, V = V, K = K, n_hyp = n_hyp, C_rowid = C_rowid, P = P, beta = as.numeric(beta), pi = as.numeric(log(pi)),
       tau2 = as.numeric(tau2), zeta = as.integer(zeta), alpha = as.numeric(alpha), C = t(Cmat), probs = probs,
       means_betas = means_betas, meansquares_betas = meansquares_betas,means_sigmas = means_sigmas, meansquares_sigmas=meansquares_sigmas, 
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
  formatChain(beta, pi, tau2, zeta, alpha, C)
}

#' @title Function \code{indEstimates}
#' @description compute independent estimates of gene specific parameters. Used for 
#' computing informative priors and initializing chain
#' @param data list, from formatData
#' @export

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
#' @export
#' 
informPriors <- function(estimates){
  V <- dim(estimates[[1]])[1]
  pr_tau2_var <- var(1/pmax(estimates[[2]],.01))
  pr_tau2_mean <- mean(1/pmax(estimates[[2]], .01))
  list(prior_mean = sapply(1:V, function(v) median(estimates[[1]][v,])),
       prior_sd = sapply(1:V, function(v) 2 * sd(estimates[[1]][v,])),
       a = pr_tau2_mean^2 / pr_tau2_var,
       b = pr_tau2_mean   / pr_tau2_var)
}

#' @title Function \code{formatControl}
#' @description Produce a formatted list for control arguments
#' @param n_iter numeric, number of post-warmup MCMC iterations
#' @param thin numeric, thinning interval
#' @param warmup numeric, number of warmup iterations
#' @param methodPi string, one of c("stickBreaking","symmDirichlet)
#' @param idx_save numeric, indices of genes for which to save draws
#' @param n_save_P numeric, number of iterations to save for random distribution
#' @param alpha_fixed logical, whether alpha is set by user, or estimated from data
#' @param slice_width numeric, initial value for slice sampler for alpha when methodPi="symmDirichlet". Tuned during warmup.
#' @param max_steps numeric, maximum number of steps in slice sampler
#' @return a named list
#' @export
#' 
formatControl <- function(n_iter, thin, warmup, methodPi="stickBreaking", idx_save=1, n_save_P=1, alpha_fixed=F, slice_width=1, max_steps=100){
  stopifnot(n_iter>=1, thin>=1, warmup>=1, methodPi %in% c("stickBreaking","symmDirichlet"), all(idx_save>=1), n_save_P>=1, slice_width>0, max_steps>1, max_steps<1000)
  out <- list(n_iter = as.integer(n_iter),
       thin = as.integer(thin),
       warmup = as.integer(warmup),
       methodPi = as.character(methodPi),
       idx_save = as.integer(idx_save-1),
       n_save_P = as.integer(n_save_P),
       alpha_fixed = as.logical(alpha_fixed),
       slice_width = as.numeric(slice_width),
       max_steps = as.integer(max_steps))
  class(out) <- "formattedControlObj"
  out
}


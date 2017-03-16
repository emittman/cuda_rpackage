#' @title Function \code{RgetDevice}
#' @description Get the index of the current GPU.
#' 
#' @export
#' @return Number of CUDA-capable GPUs.
RgetDevice = function(){
  .Call("RgetDevice", PACKAGE = "cudarpackage")
}

#' @title Function \code{RgetDeviceCount}
#' @description Get the number of CUDA-capable GPUs.
#' 
#' @export
#' @return Number of CUDA-capable GPUs.
RgetDeviceCount = function(){
  .Call("RgetDeviceCount", PACKAGE = "cudarpackage")
}

#' @title Function \code{RsetDevice}
#' @description Set the GPU for running the current MCMC chain.
#' 
#' @export
#' @return Integer index of the current CUDA-capable GPU, which is >= 0 and 
#' < number of devices.
#' @param device Index of the GPU to use. Must be an integer from 0 to number of GPUs - 1.
RsetDevice = function(device){
  .Call("RsetDevice", PACKAGE = "cudarpackage", device)
}

#' @title Function \code{Rmy_reduce}
#' @description reduce a vector on the device
#' 
#' @export
#' @return Double sum of vector elements
#' @param vec vector of doubles
Rmy_reduce = function(vec){
  .Call("Rmy_reduce", PACKAGE = "cudarpackage", vec)
}

#' @title Function \code{Rsummary}
#' @description summarize clusters
#' 
#' @export
#' @return List a matrix with sums of rows by cluster, table of occupancy counts
#' @param zeta cluster ids 0,...,K
#' @param yty numeric length G
#' @param ytx matrix V * G
#' @param K integer

Rsummary = function(zeta, yty, xty, K){
  if(length(zeta) != length(yty) | length(zeta) != ncol(xty)) stop("input dimensions don't match ")
  out <-.Call("Rsummary2", as.integer(zeta), as.numeric(yty), as.numeric(t(xty)), as.numeric(xty),
              as.integer(length(zeta)), as.integer(nrow(xty)), as.integer(K))
  names(out) <- c("num_occ","yty_sum","ytx_sum","xty_sum")
  return(out)
}

#' @title Function \code{Rchol_multiple}
#' @description cholesky factorize many matrices in parallel
#' 
#' @export
#' @return array of matrices in  with lower cholesky factor in lower triangle
Rchol_multiple = function(array){
  d <- dim(array)
  out <- .Call("Rchol_multiple", as.numeric(array), as.integer(d[1]), as.integer(d[3]))
  dim(out) <- d
  return(out)
}

#' @title Function \code{Rconstruct_prec}
#' @description get conditional precision matrix for cluster-specific parameters
#' 
#' @export
#' @return array of matrices
#' @param data constructed with formatData
#' @param priors constructed with formatPriors
#' @param chain constructed with formatChain
Rconstruct_prec = function(data, priors, chain, verbose=0){
  out <- .Call("Rconstruct_prec", data, priors, chain, verbose)
  dim(out) <- c(data$V, data$V, priors$K)
  return(out)
}

#' @title Function \code{Rbeta_rng}
#' @description draw beta random variables using curand device API
#' @export
#' @param a shape1 parameter
#' @param b shape2 parameter
Rbeta_rng = function(a, b){
  seed <- as.integer(sample(1e5, 1))
  out <- .Call("Rbeta_rng", seed, as.numeric(a), as.numeric(b))
  return(data.frame(a=a, b=b, x=out))
}

#' @title Function \code{Rquad_form_multi}
#' @description compute t(x_i)A(x_i) for i=1,...,n
#' @export
#' @param A d by dim matrix
#' @param x d*n matrix
#' @param n number of vectors
#' @param dim dimension of each vector
Rquad_form_multi = function(A, x, n, d){
  out <- .Call("Rquad_form_multi", as.numeric(A), as.numeric(x), as.integer(n), as.integer(d))
  return(out)
}

#' @title Function \code{Rdevice_mmultiply}
#' @description compute C = t(A) %*% B
#' @export
#' @param A k*m
#' @param B k*n
Rdevice_mmultiply = function(A, B){
  m <- as.integer(dim(A)[2])
  n <- as.integer(dim(B)[2])
  k <- as.integer(dim(A)[1])
  if(k != dim(B)[1]) stop("dimensions incorrect")
  out <- .Call("Rdevice_mmultiply", as.numeric(A), as.numeric(B), k, m, k, n)
  dim(out) <- c(m,n)
  return(out)
}

#' @title Function \code{mcmc}
#' @description Sample from posterior of DPMM
#' @export
#' @param data list, use formatData
#' @param priors list, use formatPrior
#' @param weightMethod Specifies the model for the weights of the
#' unknown mixture. The default is "stickBreaking" and the other option
#' is "symmDirichlet"
#' @param chain list, use formatChain
#' @param n_iter int
#' @param idx_save int vector, which genes to save (0-indexed)
#' @param thin int
#' @param n_save_P int
#' @param C numeric matrix, contrasts
#' @param verbose int, higher verbosity -> more printing
mcmc <- function(data, priors, weightMethod = "stickBreaking", chain = NULL, n_iter, idx_save, thin, n_save_P, C = NULL, verbose=0){
  if(!(data$V == length(priors$mu_0))) stop("Dimensions of prior mean don't match design matrix!")
  if(!(data$G >= priors$K)) stop("G must be <= K!")
  if(n_save_P>n_iter) stop("n_save_P must be < n_iter!")
  if(is.null(chain)){
    chain <- initChain(priors, data$G, C)
  }
  if(!(data$V == chain$V)) stop("data$V != chain$V")
  if(!(max(idx_save) < data$G)) stop("idx_save should use 0-indexing")
  methodPi <- switch(weightMethod,
                     "stickBreaking" = as.integer(0),
                     "symmDirichlet" = as.integer(1))
  seed <- as.integer(sample(1e6, 1))
  n_iter <- as.integer(n_iter)
  n_save_P <- as.integer(n_save_P)
  thin <- as.integer(thin)
  verbose <- as.integer(verbose)
  
  
  out <- .Call("Rrun_mcmc", data, priors, methodPi, chain, n_iter, n_save_P, as.integer(idx_save),
               thin, seed, verbose)
  
  names(out[[1]]) <- c("beta", "tau2", "P", "max_id", "num_occupied")
  dim(out[[1]][['beta']]) <- c(data$V, length(idx_save), ceiling(n_iter/ thin))
  dimnames(out[[1]][['beta']]) <- list(v=1:data$V, g=idx_save+1, iter=1:ceiling(n_iter/ thin))
  out[[1]][['beta']] <- aperm(out[[1]][['beta']], c(1,3,2))
  dim(out[[1]][['tau2']]) <- c(length(idx_save), ceiling(n_iter/ thin))
  dimnames(out[[1]][['tau2']]) <- list(g=idx_save+1, iter=1:ceiling(n_iter/ thin))
  dim(out[[1]][['P']]) <- c(priors$K, data$V+2, n_save_P)
  dimnames(out[[1]][['P']]) <- list(k=1:priors$K,
                                par = c("pi", sapply(1:data$V, function(v){
                                    paste(c("beta[",v,"]"), collapse="")
                                  }), "tau2"),
                                iter=1:n_save_P)
  names(out[[1]][['max_id']]) <- as.character(1:n_save_P)
  names(out[[1]][['num_occupied']]) <- as.character(1:n_save_P)
  names(out[[2]]) <- c("probs","means","meansquares")
  dim(out[[2]][['probs']]) <- c(chain$n_hyp, data$G)
  dimnames(out[[2]][['probs']]) <- list(hyp = 1:chain$n_hyp, g = 1:data$G) 
  dim(out[[2]][['means']]) <- c(data$V, data$G)
  dimnames(out[[2]][['means']]) <- list(v = 1:data$V, g = 1:data$G)
  dim(out[[2]][['meansquares']]) <- c(data$V, data$G)
  dimnames(out[[2]][['meansquares']]) <- list(v = 1:data$V, g = 1:data$G)
  out
}

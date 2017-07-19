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
  out <- .Call("Rconstruct_prec", data, priors, chain, as.integer(verbose))
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
Rquadform_multipleK = function(A, x, n, d){
  out <- .Call("Rquadform_multipleK", as.numeric(A), as.numeric(x), as.integer(n), as.integer(d))
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
#' @param data list, use \code{\link{formatData}}
#' @param priors list, use \code{\link{formatPrior}}
#' @param control list, see \code{\link{formatControl}}
#' @param chain list, optional, use \code{\link{formatChain}}
#' @param C numeric matrix, optional, contrasts
#' @param estimates list, optional, use \code{\link{indEstimates}}
#' @param verbose numeric
mcmc <- function(data, priors, control, chain = NULL, C = NULL, estimates=NULL, verbose=0){
  if(!("formattedControlObj") %in% attr(control, "class")){
    control <- do.call(formatControl, control)
  }
  stopifnot(data$V == length(priors$mu_0))
  stopifnot(data$V == chain$V, max(control$idx_save)<data$G)
  # if(!(data$G >= priors$K)) stop("G must be <= K!")
  if(n_save_P>n_iter) stop("n_save_P must be < n_iter!")
  if(is.null(chain)){
    chain <- initChain(priors, data$G, C, estimates)
    if(!control$alpha_fixed & control$methodPi == "symmDirichlet"){
      if(is.null(control$slice_width)){
        message("No value provided for slice_width, but alpha_fixed = F and methodPi = 'symmDirichlet'!\t Defaulting to 1.0")
        chain$slice_width <- 1.0
      } else{
        chain$slice_width <- as.numeric(control$slice_width)
        chain$max_steps <- as.integer(control$max_steps)
      }
    }
  }
  
  methodPi <- switch(control$methodPi,
                     "stickBreaking" = as.integer(0),
                     "symmDirichlet" = as.integer(1))
  if(methodPi == 0){
    methodAlpha <- ifelse(control$alpha_fixed, as.integer(0), as.integer(1))
  } else{
    methodAlpha <- ifelse(control$alpha_fixed, as.integer(0), as.integer(2))
  }
  
  seed <- as.integer(sample(1e6, 1))
  verbose <- as.integer(verbose)

  out <- .Call("Rrun_mcmc", data, priors, methodPi, methodAlpha, chain, control$n_iter, control$n_save_P, as.integer(idx_save),
               control$thin, seed, verbose, control$warmup)
  
  # Format the output
  if(alpha_fixed) gnames <- c("beta", "tau2", "P", "max_id", "num_occupied")
  if(!alpha_fixed) gnames <- c("beta", "tau2", "P", "max_id", "num_occupied", "alpha")
  names(out[[1]]) <- gnames
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
  out[[1]][['P']][,"pi",] <- exp(out[[1]][['P']][,"pi",])
  
  names(out[[2]]) <- c("probs","means_betas","meansquares_betas","means_sigmas","meansquares_sigmas")
  
  dim(out[[2]][['probs']]) <- c(chain$n_hyp, data$G)
  dimnames(out[[2]][['probs']]) <- list(hyp = 1:chain$n_hyp, g = 1:data$G) 
  dim(out[[2]][['means_betas']]) <- c(data$V, data$G)
  dimnames(out[[2]][['means_betas']]) <- list(v = 1:data$V, g = 1:data$G)
  dim(out[[2]][['meansquares_betas']]) <- c(data$V, data$G)
  dimnames(out[[2]][['meansquares_betas']]) <- list(v = 1:data$V, g = 1:data$G)
  dim(out[[2]][['means_sigmas']]) <- data$G
  dimnames(out[[2]][['means_sigmas']]) <- list(g = 1:data$G)
  dim(out[[2]][['meansquares_sigmas']]) <- data$G
  dimnames(out[[2]][['meansquares_sigmas']]) <- list(g = 1:data$G)
  
  names(out[[3]]) <- c("beta", "tau2", "pi", "alpha")
  dim(out[[3]][['beta']]) <- c(data$V, priors$K)
  names(out) <- c("samples","summaries","state", "samp_time")
  out
}

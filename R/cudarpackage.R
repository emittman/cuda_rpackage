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
#' @param xtx (scaled) data dim-by-dim precision matrix
#' @param Mk cluster occupancy
#' @param lambda prior precision
#' @param tau error precision
Rconstruct_prec = function(xtx, Mk, lambda, tau, K, V){
  out <- .Call("Rconstruct_prec", as.numeric(xtx), as.integer(Mk), as.numeric(lambda),
        as.numeric(tau), as.integer(K), as.integer(V))
  dim(out) <- c(V, V, K)
  return(out)
}

#' @title Function \code{Rbeta_rng}
#' @description draw beta random variables using curand device API
#' @export
#' @param a shape1 parameter
#' @param b shape2 parameter
Rbeta_rng = function(a, b){
  out <- .Call("Rbeta_rng", as.numeric(a), as.numeric(b))
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
  
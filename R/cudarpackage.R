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
#' @param all matrix of ints (data)
#' @param key vector of ints
#' @param num_clusts integer
Rsummary = function(all, key, num_clusts){
  out <-.Call("summary_stats", t(all), as.integer(key), as.integer(num_clusts))
  out[[2]] <- matrix(out[[2]], num_clusts, nrow(all))
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

#'  @title Function \code{Rquad_form_multi}
#' @description compute t(x_i)A(x_i) for i=1,...,n
#' @export'
#'  @param A d by dim matrix
#'  @param x d*n matrix
#'  @param n number of vectors
#'  @param dim dimension of each vector
Rquad_form_multi = function(A, x, n, d){
  out <- .Call("Rquad_form_multi", as.numeric(A), as.numeric(x), as.integer(n), as.integer(d))
  return(out)
}

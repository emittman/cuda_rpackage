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
  out[[2]] <- matrix(out[[2]], nrow(all), num_clusts)
  return(out)
}
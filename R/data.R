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

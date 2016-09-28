#This should get replaced by a testthat function

x <- matrix(sample(1:8, 12, replace=T), 3, 4)
key <- c(0, 1, 1, 2)
num_clust <- 4
output <- Rsummary(x, key, num_clust)

(output)
#first elt should be "[1, 2, 1, 0]"
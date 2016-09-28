#This should get replaced by a testthat function

x <- matrix(sample(1:8, 12, replace=T), 3, 4)

Rsummary(x, c(0, 1, 1, 2), 4)

#first elt should be "[1, 2, 1, 0]"
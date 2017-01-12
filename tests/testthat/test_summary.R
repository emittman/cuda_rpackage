context("Testing cluster summaries")

G <- 100
N <- 6
V <- 3
K <- 100

zeta <- factor(sample(K, G, replace=T), levels=1:K)
X <- matrix(rnorm(N*V*V), N*V, V)
yt <- matrix(rnorm(V*N*G), G, N*V)
yty <- apply(yt, 1, function(g) crossprod(g))
xty <- sapply(1:nrow(yt), function(r) yt[r,] %*% X)

require(plyr)
require(dplyr)
sums_df <- data.frame(zeta=zeta, yty = yty, t(xty)) %>%
  arrange(zeta) %>%
  tidyr::gather(key = v, value = value, -1) %>%
  ddply(.(zeta, v), summarise,
        sum = sum(value)) %>%
  tidyr::spread(key = v, value = sum)
#Mk <- as.integer(table(zeta))

outputR <- list(num_occ = length(unique(zeta)), yty_sum = sums_df$yty, ytx_sum = as.matrix(sums_df[,sapply(1:V, function(v) paste(c("X",v), collapse=""))]))
outputR$xty_sum <- t(outputR$ytx_sum)

outputC <- Rsummary(zeta, yty, xty, K)
  
test_that("Results are equal", {
  expect_true(all(unlist(sapply(1:4, function(i) sum(abs(outputR[[i]] - outputC[[i]]))<1e-6))
))
})

  
#first elt should be "[1, 2, 1, 0]"
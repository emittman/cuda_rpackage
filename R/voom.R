library(dplyr)
data <- Paschold2012::Paschold2012

# # Subset for debugging
# ID <- unique(data$GeneID)[1:9]
# data <- subset(data, GeneID %in% ID)

wide = data %>%
  mutate(genotype_replicate = paste(genotype,replicate,sep="_")) %>%
  select(GeneID, genotype_replicate, total) %>%
  tidyr::spread(genotype_replicate, total)

nonzero_id <- which(!(rowSums(wide[,-1]) == 0))

wide = wide[nonzero_id,]

R <- as.numeric(colSums(wide[,-1]))

log_cpm <- t(apply(wide[,-1], 1, function(row){
  log((row + 0.5)/(R + 1) * 1e6, base=2)
}))

group <- c(rep(1,4), rep(3,4), rep(2,4), rep(4,4))
X <- matrix(c(1, 1, 0, 0,
              1, -1, 0, 0,
              1, 0, 1, 1,
              1, 0, 1, -1), 4, 4, byrow=T)[group,]

# ols_fits <- apply(log_cpm, 1, function(y){
#   fit <- lm(y ~ 0 + X)
#   list(beta = coef(fit), sigma = sigma(fit))
# })
# 
# mus <- sapply(ols_fits, function(e) X %*% e$beta)
# sigmas <- sapply(ols_fits, function(e) e$sigma)
# avg_log_cpm <- drop(apply(log_cpm, 1, mean))
# Rtilde <- exp(mean(log(R+1)))
# avg_log_count <- avg_log_cpm + log(Rtilde, base=2) + log(1e6, base=2) 
# 
# lo <- loess(sqrt(sigmas) ~ avg_log_count, surface="direct")
# 
# #plot loess fit
# library(ggplot2)
# ord <- order(avg_log_count)
# data.frame(avg_log_count = avg_log_count[ord],
#            sqrt_sigma = sqrt(sigmas)[ord],
#            lo_pred = predict(lo)[ord]) %>%
#   ggplot(aes(x=avg_log_count, y = sqrt_sigma)) + geom_point() +
#   geom_line(aes(y = lo_pred), color = "red")
# 
# fitted_counts <- t(mus + log(rep(R,times=ncol(mus)) + 1, base=2) + log(1e6, base=2))
# prec_weights <- matrix(predict(lo, newdata = as.numeric(fitted_counts)),
#                        nrow(fitted_counts),
#                        ncol(fitted_counts)) ^ -4
# 
# #plot distribution of precision weights
# data.frame(x = as.numeric(prec_weights)) %>%
#   ggplot(aes(x)) + geom_histogram(binwidth=5) + xlim(c(0,100))

#to pass to microarray pipeline
voom_data <- list(y = log_cpm, w = prec_weights)


library(limma)
voom_out <- voomWithQualityWeights(wide[,-1],
                                   design=X,
                                   normalization="none", plot=TRUE)
str(voom_out)

hist(voom_out[[2]])


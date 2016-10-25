library(ggplot2)

par = expand.grid(a=c(.5, 10), b=c(.5, 1))
rep_par = rep(par, each=1000)

x <- Rbeta_rng(rep_par$a, rep_par$b)

plot_list <- NULL
for(i in 1:4){
  plot_list[[i]] <- x[which(x)]
}
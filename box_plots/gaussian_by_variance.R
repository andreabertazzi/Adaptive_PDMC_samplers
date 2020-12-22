library(ggplot2)
library(gridExtra)
filename_csv = "gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0_vars.csv"
df = read.csv(filename_csv)

# Box plots for ZZS vs Adaptive ZZS
df = df[df$sampler != "BPS"& df$sampler != "Adaptive BPS (full,adap)"& df$sampler != "Adaptive BPS (full,fixed)"& df$sampler != "Adaptive BPS (off,adap)"& df$sampler != "Adaptive BPS (diag,adap)"& df$sampler != "Adaptive BPS (diag,fixed)",]
df$variance <- as.factor(df$variance)

p1 <- ggplot(data=df, aes(x=variance, y=ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Variance" ,y="ESS/sec") +
  scale_y_log10() +
  #scale_y_log10(limits=c(10^(-1),10^(6.5))) +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=1))  

show(p1)


filename_csv = "gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0_vars.csv"
df = read.csv(filename_csv)
df = df[df$sampler != "ZigZag"& df$sampler != "Adaptive ZigZag (full)"& df$sampler != "Adaptive ZigZag (diag)",]
df$variance <- as.factor(df$variance)
df$sampler<- factor(df$sampler, c("BPS", "Adaptive BPS (diag,fixed)","Adaptive BPS (full,fixed)","Adaptive BPS (full,adap)","Adaptive BPS (diag,adap)","Adaptive BPS (off,adap)"))


p2 <- ggplot(data=df, aes(x=variance, y=ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Variance" ,y="ESS/sec") +
#  scale_y_log10(limits=c(10^2,10^(8))) +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25))  
show(p2)

g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}
mylegend <- g_legend(p1)
p3 <- grid.arrange(arrangeGrob(p1 + theme(legend.position="none"),
                               p2 + theme(legend.position="none"),
                               nrow=1, ncol=2),
                   mylegend, nrow=2,heights=c(10, 1))
show(p3)

## Alternative plots
filename_csv = "gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv"
df = read.csv(filename_csv)
df$sampler<- factor(df$sampler, c("ZigZag","Adaptive ZigZag (diag)","Adaptive ZigZag (full)", "BPS", "Adaptive BPS (diag,fixed)","Adaptive BPS (full,fixed)","Adaptive BPS (full,adap)","Adaptive BPS (diag,adap)","Adaptive BPS (off,adap)"))


p1 <- ggplot(data=df, aes(x=sampler, y=min_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Sampler" ,y="Minimum ESS/sec") +
  scale_y_log10() +
  #scale_y_log10(limits=c(10^(-1),10^(6.5))) +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        axis.text.x=element_blank(),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=3))  

show(p1)

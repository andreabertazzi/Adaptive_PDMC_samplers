library(ggplot2)
library(gridExtra)
filename_csv = "gaussian-results-by-correlation-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv"
df = read.csv(filename_csv)

# Box plots for ZZS vs Adaptive ZZS
df = df[df$sampler != "BPS"& df$sampler != "Adaptive BPS (full,adap)"& df$sampler != "Adaptive BPS (full,fixed)"& df$sampler != "Adaptive BPS (off,adap)",]

df$correlation <- as.factor(df$correlation)
df$sampler<- factor(df$sampler, c("ZigZag", "Adaptive ZigZag (full)"))
# df$sampler<- factor(df$sampler, c("ZigZag", "BPS", "Adaptive ZigZag (full)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)","Adaptive BPS (off,adap)"))

p1 <- ggplot(data=df, aes(x=correlation, y=avg_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Correlation" ,y="Average ESS/sec") +
  scale_y_log10() +
  #scale_y_log10(limits=c(10^(-1),10^(6.5))) +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=1))  +
  scale_fill_manual( values=c("#FF6633", "#339900"))

show(p1)


p2 <- ggplot(data=df, aes(x=correlation, y=squared_radius_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Correlation" ,y="ESS/sec (Radius)") +
#  scale_y_log10(limits=c(10^2,10^(8))) +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25))  +
  scale_fill_manual( values=c("#FF6633", "#339900"))

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

## Box plots for BPS vs ABPS
df = read.csv(filename_csv)
df = df[df$sampler != "ZigZag"& df$sampler != "Adaptive ZigZag (full)",]

df$correlation <- as.factor(df$correlation)
df$sampler<- factor(df$sampler, c("BPS","Adaptive BPS (off,adap)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)"))

p1 <- ggplot(data=df, aes(x=correlation, y=avg_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Correlation" ,y="Average ESS/sec") +
  scale_y_log10() +
  #scale_y_log10(limits=c(10^(-1),10^(6.5))) +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=1))  +
  scale_fill_manual( values=c( "gold2", "darkorange","#3399FF","springgreen3"))

show(p1)


p2 <- ggplot(data=df, aes(x=correlation, y=squared_radius_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Correlation" ,y="ESS/sec (Radius)") +
  #  scale_y_log10(limits=c(10^2,10^(8))) +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25))  +
  scale_fill_manual( values=c( "gold2", "darkorange","#3399FF","springgreen3"))

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




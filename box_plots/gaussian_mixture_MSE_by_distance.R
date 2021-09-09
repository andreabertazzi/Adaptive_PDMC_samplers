library(ggplot2)
library(gridExtra)
setwd("~/Dropbox/PhD_TUDelft/MY PAPERS/Adaptive PDMC/Data")
filename_csv = "gaussian-mixture-MSE-by-distance-dimension-30-with-correlation-0.25-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0.2-horizon-100000.0.csv"
df = read.csv(filename_csv)

# Compute the ESS
distances <- c(0,0.5,1,1.5,2,2.5)
f <- function(x) 1+0.25*x^2
vars <- f(distances)
vars_radius <- c(167,302,892,2425,5715,12109)
esspersec <- c()
esspersec_rad <- c()
# for (dis in distances){ 
for (i in 1:6){ 
  indeces <- df$distance==distances[i]
  esspersec <- c(esspersec,vars[i]/(df$mse_avg[indeces] * df$runtime[indeces]))
  esspersec_rad <- c(esspersec_rad,vars_radius[i]/(df$mse_squared_radius[indeces] * df$runtime[indeces]))
  }
df$avg_ess_per_sec <- esspersec
df$ess_per_sec_radius <- esspersec_rad
  
# Box plots for ZZS vs Adaptive ZZS
df = df[df$sampler != "BPS"& df$sampler != "Adaptive BPS (full,adap)"& df$sampler != "Adaptive BPS (full,fixed)"& df$sampler != "Adaptive BPS (off,adap)",]

df$distance <- as.factor(df$distance)
df$sampler<- factor(df$sampler, c("ZigZag", "Adaptive ZigZag (full)"))
# df$sampler<- factor(df$sampler, c("ZigZag", "BPS", "Adaptive ZigZag (full)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)","Adaptive BPS (off,adap)"))

p1 <- ggplot(data=df, aes(x=distance, y=avg_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Distance" ,y="Average ESS/sec (Mean)") +
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


p2 <- ggplot(data=df, aes(x=distance, y=ess_per_sec_radius, fill=sampler)) + 
  geom_boxplot() + labs(x = "Distance" ,y="ESS/sec (Radius)") +
#  scale_y_log10(limits=c(10^2,10^(8))) +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25))  +
  scale_fill_manual( values=c("#FF6633", "#339900"))

show(p2)

# Save with dimension 15x9
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
filename_csv = "onlyBPS-gaussian-mixture-MSE-by-distance-dimension-30-with-correlation-0.25-with-refresh-1.0-discrstep-0.5-timeadaps-10000.0-discrESS-0.2-horizon-400000.0.csv"
df = read.csv(filename_csv)
distances <- c(0,0.5,1,1.5,2,2.5)
f <- function(x) 1+0.25*x^2
vars <- f(distances)
vars_radius <- c(167,302,892,2425,5715,12109)
esspersec <- c()
esspersec_rad <- c()
# for (dis in distances){ 
for (i in 1:6){ 
  indeces <- df$distance==distances[i]
  esspersec <- c(esspersec,vars[i]/(df$mse_avg[indeces] * df$runtime[indeces]))
  esspersec_rad <- c(esspersec_rad,vars_radius[i]/(df$mse_squared_radius[indeces] * df$runtime[indeces]))
}
df$avg_ess_per_sec <- esspersec
df$ess_per_sec_radius <- esspersec_rad

df = df[df$sampler != "ZigZag"& df$sampler != "Adaptive ZigZag (full)"& df$sampler != "Adaptive BPS (full,adap)"& df$sampler != "Adaptive BPS (off,adap)",]

df$distance <- as.factor(df$distance)
df$sampler<- factor(df$sampler, c("BPS","Adaptive BPS (full,fixed)"))

p1 <- ggplot(data=df, aes(x=distance, y=1/avg_mse_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Distance" ,y="Average ESS/sec (Mean)") +
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


p2 <- ggplot(data=df, aes(x=distance, y=1/squared_radius_mse_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Distance" ,y="ESS/sec (Radius)") +
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




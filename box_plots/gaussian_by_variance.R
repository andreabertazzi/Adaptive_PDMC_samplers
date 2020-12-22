library(ggplot2)
library(gridExtra)
setwd("~/Dropbox/PhD_TUDelft/Codes")
filename_csv = "gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0_vars.csv"
df = read.csv(filename_csv)


# filename1 = "gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv"
# df_old = read.csv(filename1)
# df_old = df_old[df_old$sampler!= "Adaptive ZigZag (diag)",]
# filename2 = "ONLYAZZS_gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv"
# df_diag = read.csv(filename2)
# df_total = rbind(df_old,df_diag)
# write.csv(df_total,"gaussian-results-diagonal-dimension50-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv")



# Box plots for ZZS vs Adaptive ZZS
df = df[df$sampler != "BPS"& df$sampler != "Adaptive BPS (full,adap)"& df$sampler != "Adaptive BPS (full,fixed)"& df$sampler != "Adaptive BPS (off,adap)"& df$sampler != "Adaptive BPS (diag,adap)"& df$sampler != "Adaptive BPS (diag,fixed)",]
df$variance <- as.factor(df$variance)
# df$sampler<- factor(df$sampler, c("ZigZag", "Adaptive ZigZag (full)"))
# df$sampler<- factor(df$sampler, c("ZigZag", "BPS", "Adaptive ZigZag (full)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)","Adaptive BPS (off,adap)"))

p1 <- ggplot(data=df, aes(x=variance, y=ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Variance" ,y="ESS/sec") +
  scale_y_log10() +
  #scale_y_log10(limits=c(10^(-1),10^(6.5))) +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=1))  #+
  #scale_fill_manual( values=c("#FF6633", "#339900"))
  #  scale_fill_discrete(name = "", labels = c("ABPS (ref=0.1)","BPS (ref=0.1)","ABPS (ref=1)","BPS (ref=1)","ABPS (adap)")) 
  # scale_fill_discrete(name = "", labels = c("BPS (ref=1)","Adaptive BPS (ref=1)","Adaptive BPS (adap)")) 
 # scale_fill_discrete(name = "", labels = c("Adaptive BPS (ref=1)","BPS (ref=1)")) 

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
        legend.text = element_text(size = 25))  #+
  # scale_fill_manual( values=c("#FF6633", "#339900"))
  # guides(title="Adaptive diagonal BPS vs standard BPS", fill = guide_legend(nrow=3)) #+
  #  scale_fill_discrete(name = "", labels = c("ABPS (ref=0.1)","BPS (ref=0.1)","ABPS (ref=1)","BPS (ref=1)","ABPS (adap)")) 
  # scale_fill_discrete(name = "", labels = c("BPS (ref=1)","Adaptive BPS (ref=1)","Adaptive BPS (adap)")) 
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
# df = df[df$sampler != "ZigZag"& df$sampler != "Adaptive ZigZag (full)"& df$sampler != "Adaptive ZigZag (diag)",]
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


# 
# 
# p2 <- ggplot(data=df, aes(x=correlation, y=squared_radius_ess_per_sec, fill=sampler)) + 
#   geom_boxplot() + labs(x = "Correlation" ,y="ESS/sec (Radius)") +
#   #  scale_y_log10(limits=c(10^2,10^(8))) +
#   scale_y_log10() +
#   theme_minimal()  + 
#   theme(text=element_text(size=23),
#         legend.position = "bottom",
#         legend.title = element_blank(),
#         legend.text = element_text(size = 25))  +
#   scale_fill_manual( values=c( "gold2", "darkorange","#3399FF","springgreen3"))
# # guides(title="Adaptive diagonal BPS vs standard BPS", fill = guide_legend(nrow=3)) #+
# #  scale_fill_discrete(name = "", labels = c("ABPS (ref=0.1)","BPS (ref=0.1)","ABPS (ref=1)","BPS (ref=1)","ABPS (adap)")) 
# # scale_fill_discrete(name = "", labels = c("BPS (ref=1)","Adaptive BPS (ref=1)","Adaptive BPS (adap)")) 
# 
# show(p2)
# 
# g_legend<-function(a.gplot){
#   tmp <- ggplot_gtable(ggplot_build(a.gplot))
#   leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
#   legend <- tmp$grobs[[leg]]
#   return(legend)}
# mylegend <- g_legend(p1)
# p3 <- grid.arrange(arrangeGrob(p1 + theme(legend.position="none"),
#                                p2 + theme(legend.position="none"),
#                                nrow=1, ncol=2),
#                    mylegend, nrow=2,heights=c(10, 1))
# show(p3)




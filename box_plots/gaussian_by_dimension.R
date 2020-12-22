library(ggplot2)
library(gridExtra)
filename_csv = "gaussian-results-by-dimension-20to80-correlation0.8-with-refresh-1.0-discrstep-0.5-timeadaps-2000.0-discrESS-0-horizon-100000.0.csv"
df = read.csv(filename_csv)

df$dimension <- as.factor(df$dimension)
df$sampler<- factor(df$sampler, c("ZigZag", "BPS", "Adaptive ZigZag (full)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)"))

p1 <- ggplot(data=df, aes(x=dimension, y=avg_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Dimension" ,y="Average ESS/sec") +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 24)) +
  guides(title="Adaptive diagonal BPS vs standard BPS", fill = guide_legend(nrow=1))  +
  scale_fill_manual( values=c("#FF6633","gold2","#339900", "#3399FF","springgreen3"))

show(p1)


p2 <- ggplot(data=df, aes(x=dimension, y=squared_radius_ess_per_sec, fill=sampler)) + 
  geom_boxplot() + labs(x = "Dimension" ,y="ESS/sec (Radius)") +
  scale_y_log10() +
  theme_minimal()  + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  scale_fill_manual( values=c("#FF6633","gold2","#339900", "#3399FF","springgreen3"))

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


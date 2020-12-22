# plot by number of observations
library(ggplot2)
library(gridExtra)
# setwd("...")
filename_csv = "complete-results-by-dimensions-in-2--with-1000-observations-refresh-1.0-eps-0.1discrstep0.5timeadaps2000.0discrESS0horizon100000.0.csv"
df = read.csv(filename_csv)


# changing order of the factors
# by dimensions
df$sampler<- factor(df$sampler, c("ZigZag", "BPS", "Adaptive ZigZag (full)","Adaptive BPS (full,adap)","Adaptive BPS (full,fixed)"))
df$dimension <- as.factor(df$dimension)

p1 <- ggplot(data = subset(df, !is.na(avg_ess_per_sec)), aes(x=dimension, y=avg_ess_per_sec,fill=sampler)) +
  geom_boxplot() + labs(x = "dimension" ,y="average ESS per second") + scale_y_log10() +
  theme_minimal() +
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=2)) +
  scale_alpha_continuous(range=c(0.17,1))
show(p1)


# Squared radius
p2 <- ggplot(data = subset(df, !is.na(squared_radius_ess_per_sec)), aes(x=dimension, y=squared_radius_ess_per_sec,fill=sampler)) +
geom_boxplot() + labs(x = "dimension" ,y="squared radius ESS per second") + scale_y_log10() +
  theme_minimal() +
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
#  guides(title="", fill = guide_legend(nrow=3)) +
  scale_alpha_continuous(range=c(0.17,1)) 
show(p2)


g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}
mylegend <- g_legend(p1)
p3 <- grid.arrange(arrangeGrob(p1 + theme(legend.position="none"),
                               p2 + theme(legend.position="none"),
                               nrow=2, ncol=1),
                   mylegend, nrow=2,heights=c(10, 1))
show(p3)



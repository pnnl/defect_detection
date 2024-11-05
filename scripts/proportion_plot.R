
library(ggplot2)
library(dplyr)
library(reshape2)
library(stringr)

output_dir ='/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/tritium/defect_detection/images_paper/'
filename = '/Volumes/TTP_NeuralNet/Results/new_results_sep24/small_adam_irradiated_segnet_unirradiated_ewce.csv'
# created with copy paste from output file
df = read.csv(filename)
df$Image <-replace(df$Image, df$Image=='TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif', 1)
df$Image <-replace(df$Image, df$Image=='TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif', 2)
df$Image <-replace(df$Image, df$Image=="TTP_SEM10696_C13-1-2-2-P12_005_cropped.tif", 3)
df$Image <-replace(df$Image, df$Image=='TTP_SEM2121_C13-2-5-2-P12_037_cropped.tif', 4)
df$Image <-replace(df$Image, df$Image=='TTP_SEM_m15000x_031.tif', 5)



dfa = df %>%
  group_by(Condition, Defect, Type) %>%
  summarise(n = n(),
            Mean = mean(Percentage),
            std = sd(Percentage))
dfa$SE = qnorm(.975)*dfa$std/sqrt(dfa$n)

round_df = dfa %>% mutate_if(is.numeric, round, digits=3)
write.csv(round_df,   paste0(output_dir,  'proportion_plot_smallbayesnet_EWCE_Adam_proportion_plot_average.csv'), row.names=FALSE)


#############################
### Create Visualizations ###

dodge <- position_dodge(.5)
propotion_plot = ggplot(data=dfa, aes(y = Mean, x = Condition, color=Type)) +
  geom_point(size=5, alpha = .5, position = dodge) +
  theme_bw() + ylab("Percentage of Total")  +
  
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  facet_grid(.  ~ Defect)  +
  geom_errorbar(aes(ymax=Mean+SE, ymin=Mean-SE), 
                size=1.5, width = .25, position = dodge) +
  theme(plot.title = element_text(hjust = 0.5)) + 
 scale_y_continuous(labels = function(x) paste0(x*100, "%")) 

ggsave(paste0(output_dir, '_smallbayesnet_EWCE_Adam_proportion_plot_average.pdf'), propotion_plot)








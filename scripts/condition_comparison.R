
library(ggplot2)
library(dplyr)
library(reshape2)

##################
### User Input ###
# filter by
alim_um = 0.001888

# images again with new performance metrics
imu0 = 42.8
imu1 = 42.8
imi0 = 68.63201777462022
imi1 = 68.63270777479893
imi2 = 102.94975167393883



Area = matrix(c(
  640, 1920,  #image 0 - unirradiated 
  640, 1920, #image 1 - unirradiated
  338, 1698,  #image 2 - irradiated (1698, 338, 3)
  480, 1698,  #image 3 - irradiated (1698, 480, 3)
  512, 1768),  #image 4 - irradiated  (1768, 512, 3)
        ncol=2, byrow=T)

conditions = c( "Unirradiated", "Irradiated")

output_dir ='/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/tritium/defect_detection/images_paper/'
location_files = '/Volumes/TTP_NeuralNet/Results/new_results_sep24'
scl = c(imu0,imu1, imi0,imi1,imi2)
pdens = apply(Area, 1, prod)

#####################
### Load the data ###

return_df <- function(){
  file_name = ""
  df = NULL
  for(i in conditions){
    if (i== "Unirradiated") {
      model = 'segnet'
      opt = 'Adam'
      loss = 'EWCE'
    } else {
      model = 'smallbayessegnet'
      opt = 'Adam'
      loss = 'EWCE'
    }
    
    folder = paste0(model, '_lr1e-04_',tolower(i),'_',opt,'_new_augmentation_', loss, '_5')
    filename = paste0('_qfs_projects_tritium_', tolower(i), '_',model, '_lr1e-04_',tolower(i),'_',opt,'_new_augmentation_', loss, '_5_table_defects', '.csv')
    filename_truth = paste0('_qfs_projects_tritium_', tolower(i), '_',model, '_lr1e-04_',tolower(i),'_',opt,'_new_augmentation_', loss, '_5_table_defects', '_truth.csv')

    defects = read.csv(paste(location_files,i,folder,filename, sep="/"), header=F)
    View(defects)
    
    colnames(defects) = defects[1,]
    
    defects$Pred = "Predicted"
    
    defects_truth = read.csv(paste(location_files,i,folder,filename_truth, sep="/"), header=F)
    colnames(defects_truth) = defects_truth[1,]

    defects_truth$Pred = "Truth"
    defects = rbind(defects, defects_truth)
    
    defects$x = as.numeric(defects$x)
    defects$y = as.numeric(defects$y)
    defects$Area = as.numeric(defects$area)
    defects$on_boundary = as.numeric(defects$on_boundary)
    defects$Condition = i
    defects = defects[which(!is.na(defects$x)),]

    defects$filenames <-replace(defects$filenames, defects$filenames=='TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif', 1)
    defects$filenames <-replace(defects$filenames, defects$filenames=='TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif', 2)
    defects$filenames <-replace(defects$filenames, defects$filenames=="TTP_SEM10696_C13-1-2-2-P12_005_cropped.tif", 3)
    defects$filenames <-replace(defects$filenames, defects$filenames=='TTP_SEM2121_C13-2-5-2-P12_037_cropped.tif', 4)
    defects$filenames <-replace(defects$filenames, defects$filenames=='TTP_SEM_m15000x_031.tif', 5)
    
    defects$Image = as.factor(defects$filenames)
    defects$Defect = defects$names
    defects$Type = c('ingrain','onboundary')[defects$on_boundary+1]
    df = rbind(df, defects)
    
    View(defects)
    file_name = paste0('_', tolower(i), '_',model, '_lr1e-04_', opt,'_new_augmentation_', loss, file_name)

  }
  df = df[which(!is.na(df$y)),]
  df = df[which(df$Area >alim_um), ]
  
  
  #######################
  ## Summarize results ##
  
  
  dfa = df %>%
    group_by(Condition, Image, Defect, Pred) %>%
    summarise(n = n(),
              Mean = mean(Area),
              std = sd(Area))
  
  dfa$SE = qnorm(.975)*dfa$std/sqrt(dfa$n)
  dfa$scale = (pdens/scl^2)[dfa$Image]
  dfa$Density = dfa$n/dfa$scale
  dfa$Image = as.factor(dfa$Image)
  round_dfa = dfa %>% mutate_if(is.numeric, round, digits=3)
  drops <- c("scale")
  round_dfa = round_dfa[ , !(names(round_dfa) %in% drops)]
  write.csv( round_dfa,  paste0(output_dir, '_average_area_density', file_name, '.csv'), row.names=FALSE)

  
  dfs = df %>% 
    group_by(Condition, Defect, Type, Pred) %>%
    summarise(n = n())
  
  
  dft = dfs %>% 
    group_by(Condition, Defect, Pred) %>%
    summarise(N = sum(n))

  dfp = merge(dfs, dft, by=c('Condition','Defect', 'Pred'))
  dfp$p = dfp$n/dfp$N
  dfp$SE = qnorm(.975)*sqrt(dfp$p*(1-dfp$p)/dfp$N)
  dfp$Condition <- factor(dfp$Condition, levels=c("Unirradiated",'Irradiated'))

 
  
  
  #########################
  ### Statistical Tests ###
  
  pvals = unique(dfp[c("Defect", "Pred")])

  pvals$pvalue <- NA
  for(p in unique(dfp$Pred)){
    for(d in unique(dfp$Defect)){
      test = dfp[(dfp$Type=='onboundary')&(dfp$Defect==d)&(dfp$Pred==p),]
      res = prop.test(test$n, test$N) 
      pvals$pvalue[(pvals$Defect==d & pvals$Pred==p)] = res$p.value
    }
  }
  pvals$Significant = factor(pvals$pvalue < .05, levels=c(FALSE,TRUE))

  
  dff = merge(dfp, pvals, by=c('Defect', 'Pred'))

  
  round_dff = dff %>% mutate_if(is.numeric, round, digits=3)
  write.csv(round_dff,  paste0(output_dir, '_onboundary_', file_name, '.csv'), row.names=FALSE)
  
  
  #############################
  ### Create Visualizations ###
  
  
  
  average_area = ggplot(data=dfa, aes(y = Mean, x = Condition, color=Image)) +
    geom_point(size=5) +
    theme_bw() + ylab('Average area ('~ mu ~ m^2 ~')') + 
    scale_x_discrete(guide = guide_axis(angle = 90)) +
    facet_grid(. + Pred ~ Defect ) + 
    geom_errorbar(aes(ymax=Mean+SE, ymin=Mean-SE), position = "identity", 
                  size=1.5, width = .25) 
  ggsave( paste0(output_dir, '_average_area', file_name, '.pdf'), average_area)
  
  
  
  density_plot = ggplot(data=dfa, aes(y = Density, x = Condition, color=Image)) +
    geom_point(size=5) +
    theme_bw() + ylab('Defect density (#/'~ mu ~ m^2 ~')') + 
    scale_x_discrete(guide = guide_axis(angle = 90)) +
    facet_grid(. + Pred ~ Defect) 
  print(paste0(tolower(i), '_',model, '_lr1e-04_',tolower(i),'_',opt,'_new_augmentation_', loss, '_density_plot.pdf'))
  ggsave( paste0(output_dir, '_density_plot',file_name, '.pdf'), density_plot)
  
  
  
  propotion_grain_plot = ggplot(data=dff[dff$Type=='onboundary',], aes(y = p, x = Condition, fill = Condition, color=Significant)) + 
    geom_bar(stat="identity", size=2) +
    scale_colour_discrete(drop = FALSE) + 
    theme_bw() + ylab('Proportion on grain boundary') + 
    scale_x_discrete(guide = guide_axis(angle = 90)) +
    facet_grid(.+ Pred ~ Defect ) +
    geom_errorbar(aes(ymax=p+SE, ymin=p-SE), position = "identity", 
                  size=1.5, width = .4, color='magenta') + 
    scale_fill_viridis_d() + 
    scale_y_continuous(labels = function(x) paste0(x*100, "%")) 
  ggsave( paste0(output_dir, '_propotion_grain_plot_truth', file_name, '.pdf'), propotion_grain_plot)
  
}

return_df()

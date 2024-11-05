# load all ripley files, add name to column by rex
# plot scatter plot y name, x riply, colored by if greater or less than confidence interval
# needs to be places in ripley results folder  next to to work
# ie /Volumes/ttp_neuralnet/Results/new_results_sep24/Unirradiated_Ripley/segnet_lr1e-04_unirradiated_Adam_new_augmentation_EWCE_5/ripley
from matplotlib.lines import Line2D
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os


name_type = "unirradiated"
pred_true = 'cluster_pred_True'
name_plot = name_type + "_" +  pred_true  + "_" + 'ripley_plot.png'
output_dir ='/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/tritium/defect_detection/images_paper/' 
file_dir = '/Volumes/ttp_neuralnet/Results/new_results_sep24/Unirradiated/segnet_lr1e-04_unirradiated_Adam_new_augmentation_EWCE_5/ripley'
#file_dir = '/Volumes/ttp_neuralnet/Results/new_results_sep24/Irradiated/smallbayessegnet_lr1e-04_irradiated_Adam_new_augmentation_EWCE_5/ripley'

image_dict = {'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2': '1', 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B': '2', \
              'TTP_SEM10696_C13-1-2-2-P12_005_cropped': '3', 'TTP_SEM2121_C13-2-5-2-P12_037_cropped': '4', 'TTP_SEM_m15000x_031': '5'}

convert_table = {'r': 'Clustering', 'w': 'Within Bounds', 'b': 'Dispersion'}
def get_combination(txt):
    return re.search(
        r"_new_augmentation_EWCE_5_(.*)_pred_.*?_(.*)\.tif.*", txt).group(1)

def get_image(txt):
    return re.search(
         r"_new_augmentation_EWCE_5_(.*)_pred_.*?_(.*)\.tif.*", txt).group(2)


def color_dot(row):
    if row['ripley'] > row['UCI']:
        val = 'r'
    elif row['ripley'] < row['LCI']:
        val = 'b'
    else:
        val = 'w'
    return val

import pathlib
source_files = sorted(Path(os.path.join(file_dir, pred_true )).glob('*ripley_clusters.csv'))
#list(pathlib.Path(os.path.join(output_dir, 'cluster_pred_False'))).glob('*ripley_clusters.csv')
print(source_files)
dataframes = []
for file in source_files:
    df = pd.read_csv(file)  # additional arguments up to your needs
    df['source'] = file.name
    dataframes.append(df)

df_all = pd.concat(dataframes)

df_all['combination'] = df_all['source'].apply(get_combination)
df_all['image'] = df_all['source'].apply(get_image)
df_all["color"] = df_all.apply(color_dot, axis=1)
df_all = df_all.replace({"image": image_dict})

# get values ripley
ripley_values = pd.DataFrame()
for image in df_all['image'].unique():
    for combination in df_all['combination'].unique():
        df_value = df_all.loc[(df_all["image"]==image) & (df_all["combination"]==combination)]
        # get values when color changes
        df_value["change_color"] = df_value["color"].shift(1, fill_value=df_value["color"].head(1)) != df_value["color"]
        for index, row in df_value.iterrows():
            if row["change_color"]:
                ripley_values = pd.concat([ripley_values, pd.DataFrame({
                            'Image': [int(image)], 
                            'Combination': [combination], 
                            'Type': [convert_table[row['color']]], 
                            'Radius': [round(row['# r'], 2)]
                        })], ignore_index=True)

ripley_values.sort_values(by=['Image', 'Combination', 'Radius'], inplace = True)
ripley_values.to_csv(os.path.join(output_dir, name_plot + "_" +  pred_true + 'ripley_change_values.csv'))
df_all.to_csv(os.path.join(output_dir, name_plot + "_" +  pred_true + 'ripley_plot_df.csv'))
df_all["image_comb"] = df_all['image'] + df_all["combination"]
# filter the images
num_of_comb = len(df_all['combination'].unique())

fig, ax = plt.subplots(nrows=num_of_comb, ncols=1, sharex=True)
num_combinations = len(df_all['combination'].unique())
for count, combination in enumerate(df_all['combination'].unique()):
    df_plot = df_all[df_all["combination"]==combination]
    ax[count].scatter(df_plot["# r"], df_plot["image"].astype(
    str), c=df_plot['color'], s=2)
    ax[count].set_ylabel(combination, rotation=0, labelpad = 51)
    ax[count].margins(y=.2)

ax[num_combinations-1].set_xlabel(r'$\mu m$')


# lines for keys
custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='w', lw=4)]

# Shrink current axis by 20%
box = ax[num_of_comb-1].get_position()
ax[num_of_comb-1].set_position([box.x0, box.y0, box.width, box.height])

# Put a legend to the right of the current axis "Clustering" and "Dispersion"
ax[num_of_comb-1].legend(custom_lines, ["Clustering", "Dispersion", 'Within Bounds'], bbox_to_anchor=(1.05, 1.0))

plt.savefig(os.path.join(output_dir,  name_plot), bbox_inches="tight")

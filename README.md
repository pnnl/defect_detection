# `defect_detection` Electron Microscopy Segmentation Analysis (EMSA)


`defect_detection` provides a python package and a command line interface for
sementically segmenting micrographs to detect defects, using the methods
described in


>   Karl Pazdernik, Nicole L. LaHaye, Conor M. Artman, Yuanyuan Zhu,
>   Microstructural classification of unirradiated LiAlO2 pellets by deep learning methods,
>   *Computational Materials Science*,
>   Volume 181,
>   2020,
>   109728,
>   ISSN 0927-0256,
>   https://doi.org/10.1016/j.commatsci.2020.109728.

## Running the Jupyter Demo Notebook

### Installation

 ``` 

git clone https://github.com/pnnl/defect_detection
``` 

Create environment
``` 
conda create -n ttp_env python=3.9
conda activate ttp_env
``` 
Install defect_detection
``` 
cd defect_detection
pip install -e .
```
If recieve cuda error:
```
pip uninstall torch torchvision torchaudio
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Download files/images

Download or train (see below) to create a model weight and move the model weight file (in the notebook called "segnet_lr1e-04_unirradiated_Adam_new_augmentation_EWCE_5class_model_weights.pt") to:
```
defect_detection/notebooks/data/Best_Models/
```
Move the unirradiated images to:
```
defect_detection/notebooks/Unirradiated_images
```

### Running the Notebook

Open the Jupyter Notebook application and navigate to defect_detection/notebooks/TTP_run_predictions.ipynb. The notebook will guide you through loading the model, generating predictions, and visualizing the results.

### Training

If you're interested in training a segmentation model, there are some more dependencies than for simple detection.  You should have access to several images and their segmentation label.  


[Train Script](scripts/bsn_paper_train.py)
Script to train<br>
``` 
   python bsn_paper_metrics.py combo_num save_folder model_path
``` 
>combo_num: integer to select the item from the list of combination of parameters (parameter combos in file)<br>
>save_folder: location to save the outputs<br>
>model_path: location model pt files are saved<br>
>Output: model pt file<br>

The combination of parameters can be set by editing the list of models, optimizers (opts), and loss weights (losses) within the python script. The folder of images should be set within the train script under under variable 'folder'

The training images should include a corresponding labeled image with "Labeled " added to the start of the file name




### Detection

If you have access to a trained segmentation model, and you'd like to generate a segmented image and obtain performance metrics and ripley results, use
[Performance Metrics Script](scripts/bsn_paper_metrics.py). If you'd like to calibrate the images, use
[Calibration Script](scripts/bsn_calibrate_paper.py).

For both: <br>
``` 
   python bsn_paper_metrics.py combo_num save_folder model_path
``` 
``` 
   python bsn_calibrate_paper.py combo_num save_folder model_path
``` 
>combo_num: integer to select the item from the list of combination of parametrs (parameter combos in file)<br>
>save_folder: location to save the outputs<br>
>model_path: location model pt files are located<br>


The combination of parameters can be set by editing the list of models, optimizers (opts), and loss weights (losses) within the python script. The folder of images should be set within the train script under variable `folder`.

In both python scripts, for `no_running_mean` (line 17): set to True if don't want to use mean from the training data (try if test imaging conditions are different than the training data imaging conditions)

In defect_detection/train/__init__.py, update the dict_scaling variable at line 87 with the filename and scale value in the format `filename: {'scale': [px/um]}`. If you are calculating the Ripley value, update the `r` variable at line 645 if you want to calculate Ripley with different radius values. 


[Calibration Script](scripts/bsn_calibrate_paper.py)
Script to calibrate. The model pt file should already exist.
>Output: metrics and images for calibrated image in folder named after the model pt file<br>

[Performance Metrics Script](scripts/bsn_paper_metrics.py)
Script to get all performance metrics and ripley results. The model pt file should already exist.
 >Output: performance metrics in folder named after the model pt file<br>

## Visualizing results

[Visualize Defect Comparison](scripts/condition_comparison.R)
Script to create plots to compare defects. The user will compare the 'defects.csv' files from the best models for irradiated and unirradiated images.
>alim_um: filter by this the minimum area (in um)<br>
>imu{number}:  px/um scale<br>
>Area: test image height and width<br>
>conditions: root folder name (irradiated and unirradiated) where files to process are located<br>
>output_dir: location of output plots<br>
>location_files: file path to root folders (parent folder to irradiated and unirradiated folder containing performance metrics result folders)<br>
>line 42: the parameters of the model outputs for unirradiated and irradiated models (i.e. segnet, Adam, EWCE for model, opt, loss weight)<br>
>line 75: translations of image filenames to image integers<br>

>input: root directory with output from running bsn_paper_metrics.py<br>
>output: csv files and plots for average area, density, and proportion on grain boundary plots<br>

[Visualize Proportion](scripts/proportion_plot.R)
Script to create plots to compare pixel proportions

>output_dir: location of output plots<br>
>filename: name of file with proportions<br>
>line 11: translations of image filenames to image integers<br>

>input: proportions copied from files with 'performance_metrics_short.csv' suffix from the best metrics bsn_paper_metrics.py output folders from both irradiated and unirradiated <br>
>output: proportion graphs<br>


[Visualize Ripley Results](scripts/ripley_plot.py)
Script to create plots to compare ripley results

>name_type: irradiated vs unirradiated<br>
>pred_true: 'cluster_pred_True' or 'cluster_pred_False' if ripley is on the predicted or expert-labeled image<br>
>name_plots: name of the plot<br>
>output_dir: location of output files<br>
>file_dir: location of ripley folder (input)<br>
>image_dict: translations of image filenames to image integers<br>

>input: root directory with output from running bsn_paper_metrics.py<br>
>output: plots and csv files with radius above and below the 99% thresholds<br>
## Open-source licenses

Below are the licences and copyright notices of the code we included:
- PyTorch SegNet: [https://github.com/say4n/pytorch-segnet/tree/master?tab=MIT-1-ov-file](https://github.com/say4n/pytorch-segnet/tree/master?tab=MIT-1-ov-file)
- PyTorch Multi-Class Focal Loss: [https://github.com/AdeelH/pytorch-multi-class-focal-loss?tab=MIT-1-ov-file](https://github.com/AdeelH/pytorch-multi-class-focal-loss?tab=MIT-1-ov-file)
- TopoLoss: [https://github.com/HuXiaoling/TopoLoss?tab=MIT-1-ov-file](https://github.com/HuXiaoling/TopoLoss?tab=MIT-1-ov-file)
- Astropy: [https://github.com/astropy/astropy?tab=BSD-3-Clause-1-ov-file#readme](https://github.com/astropy/astropy?tab=BSD-3-Clause-1-ov-file#readme)
- Spatstat: [https://rdrr.io/cran/spatstat/man/spatstat-package.html](https://rdrr.io/cran/spatstat/man/spatstat-package.html)
- PyTorch UNet SegNet: [https://github.com/trypag/pytorch-unet-segnet/tree/master?tab=MIT-1-ov-file](https://github.com/trypag/pytorch-unet-segnet/tree/master?tab=MIT-1-ov-file)
- PyTorch-SemSeg: [https://github.com/meetps/pytorch-semseg/tree/master?tab=MIT-1-ov-file](https://github.com/meetps/pytorch-semseg/tree/master?tab=MIT-1-ov-file)


## License

Simplified BSD
____________________________________________
Copyright 2024 Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Disclaimer

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830




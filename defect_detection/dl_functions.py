############################################################################################
# Functions for training model: Returns training results, plots, and model fit.
# Outputs network weights, performance, and predictions.
# ###########################################################################################

from .ttp_imports import *
from PIL import Image
from PIL.TiffTags import TAGS
import json
from tifffile import TiffFile
import datetime

def mean(ls):
    #ls should be a list
    return sum(ls)/len(ls)

def torch_load_data_metadata(X_train, meta_train, X_val, meta_val, Y_train, Y_val, batch_size, shuffle=True):
    
    # Convert from numpy and put into float format
    X_train = torch.from_numpy(X_train).float()
    meta_train = torch.from_numpy(meta_train).float()
    
    Y_train = torch.from_numpy(Y_train).float()
    
    X_val = torch.from_numpy(X_val).float()
    meta_val = torch.from_numpy(meta_val).float()
    Y_val = torch.from_numpy(Y_val).float()

    # Use PyTorch dataloader to make iterable dataset for training:
    train_data = data.TensorDataset(X_train, meta_train, Y_train)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    
    val_data = data.TensorDataset(X_val, meta_val, Y_val)
    val_data_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_data_loader, val_data_loader, train_data, val_data


def torch_load_data(X_train,  X_val, Y_train, Y_val, batch_size, shuffle=True):

    # Convert from numpy and put into float format
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()

    X_val = torch.from_numpy(X_val).float()
    Y_val = torch.from_numpy(Y_val).float()

    # Use PyTorch dataloader to make iterable dataset for training:
    train_data = data.TensorDataset(X_train, Y_train)
    train_data_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle)

    val_data = data.TensorDataset(X_val, Y_val)
    val_data_loader = data.DataLoader(
        val_data, batch_size=batch_size, shuffle=shuffle)

    return train_data_loader, val_data_loader, train_data, val_data

# return metadata from images. cropped images are missing the meta data, so added from json files
def return_meta_data(image_folder, filename):
    meta_dict = {}
    file_location = os.path.join(image_folder, filename)

    # to do: need to automate hot coding of catagorical values instead of hard coding, i.e. system_type_sem
    meta_list = [ "user_time", "user_date", 'system_type_sem',  'beam_hv', 'beam_spot',\
        'beam_beam_ebeam', 'beam_scan_escan', 'ebeam_hv', 'ebeam_hfw', 'ebeam_vfw', \
            'ebeam_emissioncurrent', 'escan_dwell',  'ssd_signal_bse', 'privatefei_databarheight', 'mag']
    # make dictionary to store items
    for item in meta_list:
        meta_dict[item] = 0

    # i to only load json file of metadata if 
    i = 0
    for page in TiffFile(file_location).pages:
        for tag in page.tags.values():
            i += 1
            helios_dict = {}

            #crpped images don't have meta data in tif
            #so using meta-data from original images (without _cropped appended to name)
            if filename == "TTP_SEM2121_C13-2-5-2-P12_037_cropped.tif" and  i == 1:
            
                # with open(os.path.join(image_folder, 'TTP_SEM2121_C13-2-5-2-P12_037.json'), 'w') as fp:
                #     json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM2121_C13-2-5-2-P12_037.json')) as json_file:
                    helios_dict = json.load(json_file)
                # mag is printed on image, just manually entering for now
                meta_dict["mag"] = 10000

            elif filename == "TTP_SEM10696_C13-1-2-2-P12_005_cropped.tif" and  i == 1:

                # with open(os.path.join(image_folder, 'TTP_SEM10696_C13-1-2-2-P12_005.json'), 'w') as fp:
                #     json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM10696_C13-1-2-2-P12_005.json')) as json_file:
                    helios_dict = json.load(json_file)

                # mag is printed on image, just manually entering for now
                meta_dict["mag"] = 10000
            
            elif filename == "TTP_SEM_m15000x_031.tif" and i == 1:

                # with open(os.path.join(image_folder, 'TTP_SEM_m15000x_031.json'), 'w') as fp:
                #         json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM_m15000x_031.json')) as json_file:
                    helios_dict = json.load(json_file)
                meta_dict["mag"] = 15000

 
            # elif filename == "TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif" and tag.name == 'FEI_HELIOS':

            #     with open(os.path.join(image_folder, 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.json'), 'w') as fp:
            #         json.dump(tag.value, fp)
            #     with open(os.path.join(image_folder, 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.json')) as json_file:
            #         helios_dict = json.load(json_file)

            # elif filename == "TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif" and tag.name == 'FEI_HELIOS':

            #     with open(os.path.join(image_folder, 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.json'), 'w') as fp:
            #         json.dump(tag.value, fp)
            #     with open(os.path.join(image_folder, 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.json')) as json_file:
            #         helios_dict = json.load(json_file)
            elif tag.name == 'FEI_HELIOS':
                    helios_dict = tag.value
                    # mag is printed on image, just manually entering for now
                    if filename == 'TTP_SEM_m15000x_031.tif':
                        meta_dict["mag"] = 15000
            else:
                pass

            if helios_dict != {}:
                for key, value in helios_dict.items():
                    for key2, value2 in value.items():
                        new_key = "{}_{}".format(key, key2)
                        # convert time, date, to days/min and convert categorical to hot coded values
                        if isinstance(value2, str):

                            if new_key.lower() == "user_date":
                                format_str = '%m/%d/%Y'
                                datetime_obj = datetime.datetime.strptime(value2, format_str)
                                value2 = (datetime_obj.date() - datetime.date(2017, 1, 1)).days

                            elif new_key.lower() == "user_time":
                                time_obj = datetime.datetime.strptime(value2, "%I:%M:%S %p")
                                value2 = time_obj.hour
                                
                            else:
                                new_key = "{}_{}".format(new_key, value2)
                                value2 = 1
                        # only if new_key is in the hand-made dist of important values, include it
                        if new_key.lower() in meta_dict.keys():
                            meta_dict[new_key.lower()] = value2
    meta_values = list(meta_dict.values())
    pd.DataFrame(list(zip(meta_values, meta_list)), columns = ['name', 'value']).to_csv(os.path.join(image_folder, "meta_data_" + filename + ".csv"))
    return meta_values

# return metadata from images. cropped images are missing the meta data, so added from json files


def return_scale(image_folder, filename):
    scale_dict = {'scale': 1}
    file_location = os.path.join(image_folder, filename)

    i = 0
    for page in TiffFile(file_location).pages:
        for tag in page.tags.values():
            helios_dict = {}

            #crpped images don't have meta data in tif
            #so using meta-data from original images (without _cropped appended to name)
            if filename == "TTP_SEM2121_C13-2-5-2-P12_037_cropped.tif" and  i == 0:
                
                # with open(os.path.join(image_folder, 'TTP_SEM2121_C13-2-5-2-P12_037.json'), 'w') as fp:
                #     json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM2121_C13-2-5-2-P12_037.json')) as json_file:
                    helios_dict = json.load(json_file)

            elif filename == "TTP_SEM10696_C13-1-2-2-P12_005_cropped.tif" and  i == 0:
           
                # with open(os.path.join(image_folder, 'TTP_SEM10696_C13-1-2-2-P12_005.json'), 'w') as fp:
                #     json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM10696_C13-1-2-2-P12_005.json')) as json_file:
                    helios_dict = json.load(json_file)
            
            elif filename == "TTP_SEM_m15000x_031.tif" and i == 0:
           
                # with open(os.path.join(image_folder, 'TTP_SEM_m15000x_031.json'), 'w') as fp:
                #         json.dump(tag.value, fp)
                with open(os.path.join(image_folder, 'TTP_SEM_m15000x_031.json')) as json_file:
                    helios_dict = json.load(json_file)

            elif tag.name == 'FEI_HELIOS':
                    helios_dict = tag.value

            else:
                pass
            i += 1
            if helios_dict != {}:
                # rounding as original training images has HFW rounded
                scale_dict['scale'] = helios_dict["Image"]["ResolutionX"]/(helios_dict["EBeam"]["HFW"] * 10**6)

    return scale_dict

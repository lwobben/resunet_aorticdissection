import numpy as np
import os
import json
from scipy import ndimage
import Constants


def load_v(folder_path, name_end):
    """
    input - folder_path = path of folder you want to load header info from (from that folder and subfolders)
    input - name_end = name ending of header files you wish to get info from (excluding extension)
    return - vlist = list with header info of files in folder_path and subfolders ending with name_end.v
    """
    if os.path.isdir(folder_path):
        file = getfiles(folder_path, name_end + ".v")[0]
        with open(file) as v_file:
            vcontent = json.load(v_file)
            vpath = os.path.sep.join(str(file).split(os.path.sep)[:-1])
            vcontent["dataFile"] = (
                vpath + os.path.sep + vcontent["dataFile"]
            )  # change v datefile info into full path
    else:
        raise ValueError("The input is not an existing directory")
    return vcontent


def getfiles(folder_path, name_ending):
    """
    input - folder_path = path of folder you wish to get files from (from that folder and subfolders)
    input - name_ending = name ending of files you wish to get (including extension)
    return - allfiles = list of paths of files in folder_path and subfolders that end with name_ending
    """
    listOfFiles = os.listdir(folder_path)
    allFiles = list()
    for entry in listOfFiles:
        fullPath = os.path.join(folder_path, entry)
        if os.path.isdir(fullPath):
            # if the item in folder_path is another folder (this is the case, patientfolder)
            allFiles += getfiles(fullPath, name_ending)
        elif entry.endswith(name_ending):
            allFiles.append(fullPath)
    return allFiles


def load_volume(v_info):
    """
    input - v_info = header info
    input - z = depth to start reading from
    return - vol.astype(dtype=float) = volume with Constants.NB_SLICES slices starting from depth z
    """
    data_raw = open(v_info["dataFile"], "rb")  # open raw file belonging to header
    slice_len = v_info["size"][0]
    slice_size = slice_len**2
    nb_slices = v_info["size"][2]
    if v_info["dataType"].endswith("16"):  # Int16 occupies 16-bits (2-bytes) space
        slice_size *= 2
    content = data_raw.read()
    vol = np.fromstring(content, dtype=v_info["dataType"])
    vol = np.reshape(vol, [slice_len, slice_len, nb_slices], "F")
    return vol


def norm_volume(vol):
    """
    input - vol = volume to be normalized
    return - vol = normalized volume
    """
    max, min = np.max(vol), np.min(vol)
    if max != min:
        scale = 1 / (max - min)
        vol = scale * (vol - min)
    return vol


def preprocess(input_path_scan, output_path_scan, zoomfactor):
    """
    input - input_path = input path of folder of one scan
    input - output_path = output path of folder of one scan
    """
    v_img = load_v(input_path_scan, "data_tcl")
    v_mask = load_v(input_path_scan, "mask_tcl")
    img = load_volume(v_img)
    mask = load_volume(v_mask)
    if zoomfactor != None:
        img = ndimage.zoom(
            img, zoomfactor, order=3
        )  # downsample, order = 0 (neirest neighbour)
        mask = ndimage.zoom(
            mask, zoomfactor, order=0
        )  # downsample, order = 0 (neirest neighbour)
    img = norm_volume(img)
    os.mkdir(output_path_scan)
    np.save(output_path_scan + "/img", img.astype(dtype=float))
    np.save(output_path_scan + "/mask", mask.astype(dtype=float))


def preprocess_all_scans(input_path_all, output_path_all, zoomfactor):
    """
    input - input_all = input path of folder with all scans
    input - output_all = output_path of folder with all scans
    """
    scanlist = os.listdir(input_path_all)
    scanlist.sort()
    for scan in scanlist:
        print(scan)
        preprocess(input_path_all + scan, output_path_all + scan, zoomfactor)


if __name__ == "__main__":
    zoomfactor = None
    path_org = Constants.DIR_DATA + "PATH_ORG"
    path_new = Constants.DIR_DATA + "PATH_NEW"
    preprocess_all_scans(path_org, path_new, zoomfactor)

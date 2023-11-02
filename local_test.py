import os
import pathlib
import pprint as pp
import shutil
from io import BytesIO
from os.path import basename, join

from fastai.vision.all import *
import timm
from natsort import natsorted

if platform.system() == "Windows":
    print("on Windows OS - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


def load_best_model():
    try:
        path_to_archive = r"model-resnetv2_50x1_bigtransfer_u.zip"
        best_model_name = "model-resnetv2_50x1_bigtransfer.pkl"
        shutil.unpack_archive(path_to_archive)
        best_model = load_learner(join(os.getcwd(), best_model_name), cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_b_best = "https://www.dropbox.com/scl/fi/kfgvaam338d7qfyc4y0mr/model-resnetv2_50x1_bigtransfer.pkl?dl=1"
        best_model_response = requests.get(model_b_best)
        best_model = load_learner(BytesIO(best_model_response.content), cpu=True)

    return best_model


def load_mixnet_model():
    try:
        path_to_model = r"model-mixnetXL-20epoch_u.pil"
        model = load_learner(path_to_model, cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_backup = (
            "https://www.dropbox.com/scl/fi/48ez7tzm1q7h4o5njn0q8/model-mixnetXL-20epoch.pkl?dl=1"
        )
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)

    return model


def load_dir_files(directory, req_extension=".txt", return_type="list", verbose=False):
    appr_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for prefile in f:
            if prefile.endswith(req_extension):
                fullpath = os.path.join(r, prefile)
                appr_files.append(fullpath)

    appr_files = natsorted(appr_files)

    if verbose:
        print("A list of files in the {} directory are: \n".format(directory))
        if len(appr_files) < 10:
            pp.pprint(appr_files)
        else:
            pp.pprint(appr_files[:10])
            print("\n and more. There are a total of {} files".format(len(appr_files)))

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose:
            print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict

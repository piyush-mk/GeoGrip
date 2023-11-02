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


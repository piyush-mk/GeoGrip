import time
import os
import pathlib, platform
import pprint as pp
import shutil
from io import BytesIO
from os.path import basename, join
import timm

from natsort import natsorted
import skimage
import streamlit as st
from fastai.vision.all import PILImage, load_learner, Image, platform, requests
import timm
from natsort import natsorted
from skimage import io
from skimage.transform import resize

if platform.system() == "Windows":
    print("on Windows OS - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

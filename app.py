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


def load_best_model():
    try:
        path_to_archive = r"model-resnetv2_50x1_bigtransfer_u.zip"
        best_model_name = "model-resnetv2_50x1_bigtransfer.pkl"
        shutil.unpack_archive(path_to_archive)
        best_model = load_learner(join(os.getcwd(), best_model_name), cpu=True)
    except:
        st.write("unable to load locally. downloading model file")
        model_b_best = "https://www.dropbox.com/scl/fi/kfgvaam338d7qfyc4y0mr/model-resnetv2_50x1_bigtransfer.pkl?dl=1"
        best_model_response = requests.get(model_b_best)
        best_model = load_learner(BytesIO(best_model_response.content), cpu=True)

    return best_model


def load_mixnet_model():
    try:
        path_to_model = r"model-mixnetXL-20epoch_u.pil"
        model = load_learner(path_to_model, cpu=True)
    except:
        st.write("unable to load locally. downloading model file")
        model_backup = (
            "https://www.dropbox.com/scl/fi/48ez7tzm1q7h4o5njn0q8/model-mixnetXL-20epoch.pkl?dl=1"
        )
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)

    return model


supplemental_dir = os.path.join(os.getcwd(), "info")
fp_header = os.path.join(supplemental_dir, "climb_area_examples.png")

st.title("GeoGrip: A satellite rock climbing spot detection app")
st.markdown(
    "by Piyush Mohapatra | [GitHub](https://github.com/piyush-mk)"
)

st.markdown(
    "and Kunal"
)


with st.beta_container():
    st.markdown(
        "*Welcome to our app that evaluates satellite or aerial images of the selected terrain and "
        "and determines its suitability for outdoor bouldering.*"
    )
st.markdown("---")
st.markdown("**Examples of Images in the *climb area* class**")
st.image(skimage.io.imread(fp_header))
st.markdown("---")
with st.beta_container():
    st.subheader("Test sattelite images")
    st.markdown(
        "The following images were not used for model training"
    )


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def predict(img, img_flex, use_best_model=False):
    st.image(img, caption="Image chosen to analyze", use_column_width=True)

    if use_best_model:
        model_pred = load_best_model()
    else:
        model_pred = load_mixnet_model()

    with st.spinner("model inference running..."):
        time.sleep(3)
        if not isinstance(img_flex, str):
            fancy_class = PILImage(img_flex)
            model_pred.precompute = False
            pred_class, pred_items, pred_prob = model_pred.predict(fancy_class)
        else:
            pred_class, pred_items, pred_prob = model_pred.predict(img_flex)
        prob_np = pred_prob.numpy()

    if str(pred_class) == "climb_area":
        st.balloons()
        st.subheader(
            "Area in test image is good for climbing! {}% confident.".format(
                round(100 * prob_np[0], 2)
            )
        )
    else:
        st.subheader(
            "Area in test image not great for climbing :/ - {}% confident.".format(
                100 - round(100 * prob_np[0], 2)
            )
        )


want_adv = st.checkbox("Use Advanced model (slower)")
if want_adv:
    st.markdown("*analyzing with advanced model*")
option1_text = "Use an example image"
option2_text = "Upload a custom image for analysis"
option = st.radio("Choose a method to load an image:", [option1_text, option2_text])

if option == option1_text:
    working_dir = os.path.join(os.getcwd(), "test_images")
    test_images = natsorted(
        [
            f
            for f in os.listdir(working_dir)
            if os.path.isfile(os.path.join(working_dir, f))
        ]
    )
    test_image = st.selectbox("Please select a test image:", test_images)

    if st.button("Analyze!"):
        file_path = os.path.join(working_dir, test_image)
        img = skimage.io.imread(file_path)
        img = resize(img, (256, 256))

        predict(img, file_path, want_adv)
else:
    image_file = st.file_uploader("Upload Image", type=["png", "jpeg", "jpg"])
    if st.button("Analyze!"):
        if image_file is not None:
            file_details = {
                "Filename": image_file.name,
                "FileType": image_file.type,
                "FileSize": image_file.size,
            }
            base_img = load_image(image_file)
            img = base_img.resize((256, 256))
            img = img.convert("RGB")
            predict(img, img, want_adv)
st.markdown("---")
st.subheader("How it Works:")
st.markdown(
    "**GeoGrip** uses Convolutional Neural Network (CNN) trained on a labeled dataset ("
    "approx. 3000 satellite images, each 256x256 in two classes) with two classes. More "
    "specifically, the primary model is MixNet-XL"
)

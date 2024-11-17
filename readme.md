# IE643 Class Project: Generating missing frames of a handwritten video

This repository provides tools for generating intermediate frames in videos of handwritten content. It includes dataset preparation scripts, fine-tuning steps, training processes and also a interface to try out.

---

## Training or Fine-tuning

### 1. Install Dependencies
Install all necessary dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Video Download and Preprocess
- **Original Videos**: For automatic downloading the original videos, install yt-dlp package `pip install yt-dlp`, then run the following script-
```bash
python -m yt_dlp -i https://www.youtube.com/playlist?list=PL7yFTnWQZNwAXhwViau6Km9RlTEqVXGAe
```
- **Pre-Processing**: Run the dataset processing script to prepare smaller videos for training-
```bash
python data_preprocess_smaller_vid.py
```

### 3. Processed Dataset
You can also download the already Processed Dataset from this [LINK](https://drive.google.com/drive/folders/10QZtaSINJeB1eYkHgmlbuXfXkoPKAGGH?usp=sharing).


### 4. Training
For training the model, run the **training.ipynb** notebook.

### 5. Fine-Tuning
For fine-tuning, download the original pretrained RIFE model from [LINK](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view?usp=sharing). Place the model in the **new_train_log** folder for fine-tuning. Change the 2nd last cell of **training.ipynb** notebook, and run it.

---
---

## Interface 

This interactive interface is designed for generating intermediate frames in videos of handwritten content. One can select either a video or two images as input and set the number of interpolated frames via a slider. The output is displayed as a processed video or a gallery of frames including the original frames.

### Features

- **Upload Video**: Upload a video (with low fps or removed frames) to generate intermidate interpolated frames.
- **Upload Images**: Upload two images, and it will generate any number of in between frames.
- **Interactive Slider**: Set the number (its in power of 2) of interpolated frames using the slider.
- **Output**: You can view output video or interpolated frames.


### Setup Instructions

#### Prerequisites

- Required Python packages:
  - `gradio`
  - `opencv-python`
  - `ffmpeg-python`
  - `Pillow`

#### Setup and Execute 

- Download the model from this  [LINK](https://drive.google.com/file/d/1AL_hA3o47FV6be15ODzzVEkbcwNzMzML/view?usp=sharing) and make sure that the model is in the right path, i.e., under **new_train_log** folder.
- Run the interactive python file `demo_interface.ipynb`
- A link will be given after running all the cells, you can open that link and try it.


---
---

## Best Model
The best trained model can be found in this [LINK](https://drive.google.com/file/d/1AL_hA3o47FV6be15ODzzVEkbcwNzMzML/view?usp=sharing).

---
---

## Notes
- Path Updates: Ensure all file/folder paths are correctly set before running the any scripts or notebook, you can find the path by searching "NOTE" in the files.
- Newer Codes and model: It will be updated shortly.
- Code Attribution: Most of the training code is taken from [LINK](https://github.com/hzwer/ECCV2022-RIFE).
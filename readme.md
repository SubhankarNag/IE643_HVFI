# IE643 Class Project: Generating missing frames of a handwritten video

This repository provides tools for generating intermediate frames in videos of handwritten content. It includes dataset preparation scripts, fine-tuning steps, training processes, pretrained models, and also a interface to try out.

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
You can also download the already Processed Dataset from this [LINK](https://drive.google.com/drive/folders/10QZtaSINJeB1eYkHgmlbuXfXkoPKAGGH?usp=sharing) and store it in the `dataset` folder.


### 4. Training
For training the model, run the following script, the intermidate results and the best model will be store in the `new_train_log` folder - 
```bash
python train.py
```

### 5. Testing
Almost all the model files can be found in the `models` folder, and trained model can be found in `trained_models` folder. If you want to check the performance of these models on the dataset, you need to change the `MODEL_NAME` in `choose_model.py` file and change parameters in `utils.config.py` according to the chosen model. Then run the following script and it will output PSNR for all the splits -
```bash
python test.py
```

### 5. Hyperparameters (Only Important ones)
File: **utils.config.py**
- ```num_frames = 9```        ---> Use 9 frames when using context based models, else 7 frames
- ```num_instances = 9```     ---> Max 42 or 33 instances/video incase of 7 or 9 frames respectively
- ```is_context = True```     ---> True when using context based models, else False
- ```is_finetuning = False``` ---> True if you want to finetune a model, else False  

File: **choose_model.py**
- ```MODEL_NAME = "..."```       ---> Choose any name of the available model

### The best model
The best configuration is already set, in which **41.32 PSNR** is achieved.
```bash
Training from scratch + Region Loss + More Context + Batch-Norm + Dropout + 33 instances + 9 frames
```

---
---

## Interface 

This interactive interface is designed for generating intermediate frames in videos of handwritten content. One can select either a video or two images as input and set the number of interpolated frames via a slider. The output is displayed as a processed video or a gallery of frames including the original frames.

### Features

- **Upload Video**: Upload a video (with low fps or removed frames) to generate intermidate interpolated frames.
- **Upload Images**: Upload 2/4 images, and it will generate any number of in between frames.
- **Interactive Slider**: Set the number (its in power of 2) of interpolated frames using the slider.
- **Output**: You can view output video or interpolated frames.

#### Prerequisites

- Required Python packages:
  - `gradio`
  - `opencv-python`
  - `ffmpeg-python`
  - `Pillow`

#### Setup and Execute 

- Change the `MODEL_NAME` in  `choose_model.py` file
- Run the interactive python file `demo_interface.ipynb` or `demo_interface_context.ipynb` accordingly
- After running all the cells, a link will be displayed, one can open that link and try it.
- Some sample input images or video can be found in `demo_examples` folder.


---
---

## Notes
- Path Updates: Ensure all file/folder paths are correctly set before running the any scripts or notebook.
- Code Attribution: Most of the training code is taken from [LINK](https://github.com/hzwer/ECCV2022-RIFE).
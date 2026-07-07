# Image-Recognition-Plane-vs.-Helicopter
This project demonstrates how to train a custom image classification model using [Teachable Machine by Google](https://teachablemachine.withgoogle.com/) and deploy it in a Python environment using Google Colab.
## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Steps](#project-steps)
- [Code Implementation](#code-implementation)
- [License](#license)

## Prerequisites
* A computer with an internet connection.
* A collection of airplane and helicopter images.
## Project Steps

### 1. Initialize Project
Navigate to [Teachable Machine](https://teachablemachine.withgoogle.com/), click **Get Started**, then select **Image Project** > **Standard image model**.

### 2. Define Classes
Create two distinct classes:
* **Class 1:** "Plane"
* **Class 2:** "Helicopter"
<img width="1912" height="967" alt="لقطة شاشة 2026-07-07 233202" src="https://github.com/user-attachments/assets/c74d2a30-e330-4dc4-bef2-a48cbb1803d0" />

### 3. Upload Training Data
Upload your images into their respective folders. 
> **Note:** The accuracy of your model directly depends on the quantity and variety of images you provide.
> <img width="1906" height="965" alt="لقطة شاشة 2026-07-07 215826" src="https://github.com/user-attachments/assets/082392a9-c7b5-426f-8913-7877b50f606c" />


### 4. Train and Test
Click **Train Model**. Once finished, ensure you change the input setting from **Webcam** to **File** to test with your own image.
<img width="1913" height="968" alt="لقطة شاشة 2026-07-07 215859" src="https://github.com/user-attachments/assets/686711f8-34b4-4f85-a72d-fb48ad85a911" />
After testing, the model successfully recognized the picture of the plane.

### 5. Export the Model
Click **Export Model**, select **TensorFlow** > **Keras**, and click **Download my model**. Copy the provided starter code.
<img width="1912" height="970" alt="لقطة شاشة 2026-07-07 215919" src="https://github.com/user-attachments/assets/3d52ea9f-7c53-4e96-a772-d4e51db63656" />

### 6. Setup in Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Paste the code into a cell.
<img width="1912" height="908" alt="لقطة شاشة 2026-07-07 223842" src="https://github.com/user-attachments/assets/28160f60-d0e1-4989-aac1-04a07f10622e" />

### 7. Upload Project Files
Before running the code, you must upload the necessary files to the Google Colab session:
1. Click the Files icon (folder icon) on the left-hand sidebar.
2. Click the Upload to session storage button.
3. Select and upload keras_model.h5, labels.txt, and the picture you want to test the model with (e.g., `31.jpg`).
<img width="552" height="771" alt="لقطة شاشة 2026-07-08 001421" src="https://github.com/user-attachments/assets/baec94f5-6bb9-4af2-9183-8fb9e5fb977a" />

### 8. Code Modifications
Update your code with the following changes:
* Add "import tf_keras as tk" after the third line.
* Change "load_model" to "tk.models.load_model".
* Update the image path to your test image.
* <img width="897" height="712" alt="لقطة شاشة 2026-07-08 000127" src="https://github.com/user-attachments/assets/7aff39ed-ec74-4ed3-b2fb-706ba4f8fc2d" />
* <img width="882" height="710" alt="لقطة شاشة 2026-07-08 000412" src="https://github.com/user-attachments/assets/e04a276a-2e95-4efc-b54d-e38b38d8ae0b" />
* <img width="1903" height="926" alt="لقطة شاشة 2026-07-08 001534" src="https://github.com/user-attachments/assets/e814c764-0811-4c72-a5b7-7167f7c14ddb" />

### 9. Run and Verify
After updating the code (see below), click the "Play" button.

*![Insert Image: Screenshot of the output console showing the prediction result]*

## Code Implementation

Ensure you have updated the code to include `tf_keras` as shown below:

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tf_keras as tk 

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your test image
image = Image.open("31.jpg").convert("RGB")

# Resizing and cropping
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn into numpy array
image_array = np.asarray(image)

# Normalize
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load into array
data[0] = normalized_image_array

# Predict
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

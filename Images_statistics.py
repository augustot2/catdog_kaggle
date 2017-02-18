5import matplotlib.pyplot as plt
from PIL import Image
import os
import psutil
import time
import pandas as pd

os.chdir("/home/augusto/catdog_kaggle/test")

images = os.listdir()
num_img = len(images)
num_img
#images = images[1:20]
Dim_table = pd.DataFrame(index = images, columns=["width","height"])
i = 0
for file in images:
    image = Image.open(file)
    Dim_table.ix[file,:] = image.size
    i = i +1
    print(i)
    #image.show()
    #time.sleep(.05)

Dim_table.to_csv(path_or_buf="../size_of_images.csv")

for proc in psutil.process_iter():
    if proc.name() == "display":
        proc.kill()

# convert image to array
import numpy as np
image = Image.open(images[1])

table = pd.read_csv("../size_of_images.csv")
table

height = table.ix[:,2]
height.mean()
height.min()
height.max()
height.std()
histogram_h = np.histogram(height)
plt.hist(height,bins="auto")
plt.title("height histogram")
plt.show()

width = table.ix[:,1]
width.mean()
width.min()
width.max()
width.std()
plt.hist(width,bins="auto")
plt.title("width histogram")
plt.show()

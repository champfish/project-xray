
from PIL import Image
from matplotlib import image, pyplot
import numpy as np
import os 
import sys

image1 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0115-0001.jpeg"))
image2 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0117-0001.jpeg"))
image3 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0119-0001.jpeg"))
image4 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0122-0001.jpeg"))
image5 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0125-0001.jpeg"))
image6 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0127-0001.jpeg"))
image7 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0128-0001.jpeg"))
image8 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0129-0001.jpeg"))
image9 = image.imread(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0131-0001.jpeg"))


fig, axes = pyplot.subplots(3,3)

axes[0][0].imshow(image1)
axes[0][1].imshow(image2)
axes[0][2].imshow(image3)
axes[1][0].imshow(image4)
axes[1][1].imshow(image5)
axes[1][2].imshow(image6)
axes[2][0].imshow(image7)
axes[2][1].imshow(image8)
axes[2][2].imshow(image9)

#pyplot.imshow(image)
pyplot.show()

image_data = Image.open(os.path.join(sys.path[0], "NORMAL_TRAIN/IM-0115-0001.jpeg"))
image_data = np.asarray(image_data)
print(image_data)

def crop_images(source, destination):
    height_array = []
    width_array = []
    counter = 0
    for filename in os.listdir(os.path.join(sys.path[0], source)):
        f = os.path.join(os.path.join(sys.path[0], source), filename)
        # checking if it is a file
        img = Image.open(f)
        width = img.width
        height = img.height
        height_array.append(height)
        width_array.append(width)
        if height < width:
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            cropped = img.crop((width/2 - height/2,0,width/2 + height/2,height))
            cropped.save(os.path.join(sys.path[0], f"{destination}/img{counter}.jpeg"))
        counter += 1

    # print(height_array)
    # print(width_array)

    length = len(height_array)

    height_greater_than_width = []
    for i in range(0, length):
        if height_array[i] > width_array[i]:
            height_greater_than_width.append(i)

    total_height_greater_than_width = len(height_greater_than_width)

    max_height = max(height_array)
    min_height = min(height_array)
    max_width = max(width_array)
    min_width = min(width_array)
    height_std = statistics.stdev(height_array)
    width_std = statistics.stdev(width_array)
    height_avg = sum(height_array)/len(height_array)
    width_avg = sum(width_array)/len(width_array)

    print(f"max height: {max_height}")
    print(f"min height: {min_height}")
    print(f"max width: {max_width}")
    print(f"min width: {min_width}")
    print(f"height standard dev: {height_std}")
    print(f"width standard dev: {width_std}")
    print(f"height avg: {height_avg}")
    print(f"width avg: {width_avg}")
    print(f"total height greater than width: {total_height_greater_than_width}")
    print(f"length: {length}")

def verify_cropped_images(source):
    height_array = []
    width_array = []
    counter = 0
    for filename in os.listdir(os.path.join(sys.path[0], source)):
        f = os.path.join(os.path.join(sys.path[0], source), filename)
        # checking if it is a file
        img = Image.open(f)
        width = img.width
        height = img.height
        height_array.append(height)
        width_array.append(width)
        if height != width:
            print("height not equal to width")
            return False

    # print(height_array)
    # print(width_array)

    length = len(height_array)

    height_greater_than_width = []
    for i in range(0, length):
        if height_array[i] > width_array[i]:
            height_greater_than_width.append(i)

    total_height_greater_than_width = len(height_greater_than_width)

    max_height = max(height_array)
    min_height = min(height_array)
    max_width = max(width_array)
    min_width = min(width_array)
    height_std = statistics.stdev(height_array)
    width_std = statistics.stdev(width_array)
    height_avg = sum(height_array)/len(height_array)
    width_avg = sum(width_array)/len(width_array)

    print(f"max height: {max_height}")
    print(f"min height: {min_height}")
    print(f"max width: {max_width}")
    print(f"min width: {min_width}")
    print(f"height standard dev: {height_std}")
    print(f"width standard dev: {width_std}")
    print(f"height avg: {height_avg}")
    print(f"width avg: {width_avg}")
    print(f"total height greater than width: {total_height_greater_than_width}")
    print(f"length: {length}")
    return True

def scale_down_images(source, destination):
    height_array = []
    width_array = []
    counter = 0
    for filename in os.listdir(os.path.join(sys.path[0], source)):
        f = os.path.join(os.path.join(sys.path[0], source), filename)
        # checking if it is a file

        img = Image.open(f)
        width = img.width
        height = img.height
        height_array.append(height)
        width_array.append(width)
        new_image = img.resize((128, 128))
        new_image.save(os.path.join(sys.path[0], f"{destination}/img{counter}.jpeg"))
        counter += 1

    # print(height_array)
    # print(width_array)

    length = len(height_array)

    height_greater_than_width = []
    for i in range(0, length):
        if height_array[i] > width_array[i]:
            height_greater_than_width.append(i)

    total_height_greater_than_width = len(height_greater_than_width)

    max_height = max(height_array)
    min_height = min(height_array)
    max_width = max(width_array)
    min_width = min(width_array)
    height_std = statistics.stdev(height_array)
    width_std = statistics.stdev(width_array)
    height_avg = sum(height_array)/len(height_array)
    width_avg = sum(width_array)/len(width_array)

    print(f"max height: {max_height}")
    print(f"min height: {min_height}")
    print(f"max width: {max_width}")
    print(f"min width: {min_width}")
    print(f"height standard dev: {height_std}")
    print(f"width standard dev: {width_std}")
    print(f"height avg: {height_avg}")
    print(f"width avg: {width_avg}")
    print(f"total height greater than width: {total_height_greater_than_width}")
    print(f"length: {length}")

def verify_scaled_down_images(source):
    height_array = []
    width_array = []
    counter = 0
    for filename in os.listdir(os.path.join(sys.path[0], source)):
        f = os.path.join(os.path.join(sys.path[0], source), filename)
        # checking if it is a file

        img = Image.open(f)
        width = img.width
        height = img.height
        height_array.append(height)
        width_array.append(width)
        if height != width:
            print("dimensions are not equal")
            return False
        counter += 1

    # print(height_array)
    # print(width_array)

    length = len(height_array)

    height_greater_than_width = []
    for i in range(0, length):
        if height_array[i] > width_array[i]:
            height_greater_than_width.append(i)

    total_height_greater_than_width = len(height_greater_than_width)

    max_height = max(height_array)
    min_height = min(height_array)
    max_width = max(width_array)
    min_width = min(width_array)
    height_std = statistics.stdev(height_array)
    width_std = statistics.stdev(width_array)
    height_avg = sum(height_array)/len(height_array)
    width_avg = sum(width_array)/len(width_array)

    print(f"max height: {max_height}")
    print(f"min height: {min_height}")
    print(f"max width: {max_width}")
    print(f"min width: {min_width}")
    print(f"height standard dev: {height_std}")
    print(f"width standard dev: {width_std}")
    print(f"height avg: {height_avg}")
    print(f"width avg: {width_avg}")
    print(f"total height greater than width: {total_height_greater_than_width}")
    print(f"length: {length}")
    return True

#PNEUMONIA
# crop_images("val/PNEUMONIA", "cleaned_val/PNEUMONIA")
# res = verify_cropped_images("cleaned_val/PNEUMONIA")
# if res:
#     scale_down_images("cleaned_val/PNEUMONIA", "cleaned_val/PNEUMONIA")
#     verify_scaled_down_images("cleaned_val/PNEUMONIA")
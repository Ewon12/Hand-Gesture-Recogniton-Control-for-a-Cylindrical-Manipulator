import cv2
import pandas as pd
import os
import numpy as np

def image_processed(file_path):
    # Reading the static image
    hand_img = cv2.imread(file_path)

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the image in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # Perform your custom image processing here
    # ...

    # Placeholder return value
    return np.zeros([1, 63], dtype=int)[0]

def make_csv():
    mypath = 'DATASET'
    file_name = open('dataset1.csv', 'a')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass
        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc)
                    try:
                        for id, i in enumerate(data):
                            if id == 0:
                                print(i)

                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')

                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')

    file_name.close()
    print('Data Created !!!')

if __name__ == "__main__":
    make_csv()

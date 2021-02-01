from cv2 import imread, countNonZero, cvtColor, COLOR_BGR2GRAY, IMREAD_COLOR, imwrite
from glob import glob
import numpy as np
import os

path = 'E:\\DeepLearning\\TEKNA\\TyreSemanticSegmentation\\SegmentationReconstruction\\CambioRappresentazione\\UNIVPM\\16BIT_RGBX8BIT\\v6\preprocessed\\Image\\Bridgestone'
folder = '\\train'
mask_folder = 'E:\DeepLearning\TEKNA\TyreReconstruction\dataset_mask'
no_defect_count = 0
defect_count = 0

with open('train_no_defect.txt', 'w') as train_no_defect:
    with open('train_with_defect.txt', 'w') as train_with_defect:

        for img_path in glob(path + '/*.png'):
            image = imread(img_path, IMREAD_COLOR)
            if countNonZero(cvtColor(image, COLOR_BGR2GRAY)) == 0:
                continue
            label = imread(img_path.replace('Image', 'Label'), 0)

            try:
                label[np.where(label == 4)] = 0
            except:
                print(img_path)

            if countNonZero(label) == 0:
                # immagine che non ha difetti
                train_no_defect.write(img_path + '\n')
                os.makedirs(mask_folder, exist_ok=True)
                imwrite(mask_folder + '\\' + img_path.split('\\')[-1], label)
                no_defect_count += 1
                continue
            elif len(np.where(label == 1)[0]) != 0 and len(np.where(label == 2)[0]) == 0 and len(np.where(label == 3)[0]) == 0:
                train_no_defect.write(img_path + '\n')
                os.makedirs(mask_folder, exist_ok=True)
                imwrite(mask_folder + '\\' + img_path.split('\\')[-1], label)
                no_defect_count += 1
                continue
            mask = np.zeros(label.shape, dtype=np.int8)
            mask[np.where(label == 2)] = 1
            mask[np.where(label == 3)] = 1
            #mask[np.where(label == 4)] = 255
            train_with_defect.write(img_path + '\n')
            defect_count += 1
            os.makedirs(mask_folder, exist_ok=True)
            imwrite(mask_folder + '\\' + img_path.split('\\')[-1], mask * 255)

print('No defect count:', no_defect_count)
print('Defect count:', defect_count)

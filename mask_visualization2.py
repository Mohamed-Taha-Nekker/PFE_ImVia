import cv2 as cv
import numpy as np
import os
import nibabel
from PIL import Image

images_path = 'ACDC_new/images/'
masks_path = 'ACDC_new/masks/'

images = sorted(os.listdir(images_path))
masks = sorted(os.listdir(masks_path))

for i, j in zip(images, masks):

    OUTPUT_NAME = i + "_output.jpg"

    image_filename = os.path.join(images_path, i)
    mask_filename = os.path.join(masks_path, j)

    img = nibabel.load(image_filename)
    msk = nibabel.load(mask_filename)

    img = img.get_fdata()
    msk = msk.get_fdata()

    for k in range(img.shape[2]):

        im = np.float32(img[:,:,k])
        ms = np.float32(msk[:,:,k])

        colored_image = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
        colored_mask = cv.cvtColor(ms, cv.COLOR_GRAY2RGB)

        for i in range(colored_mask.shape[0]):
            for j in range(colored_mask.shape[1]):
                if (colored_mask[i, j] > (80, 80, 80)).all():
                    if (colored_mask[i, j] > (230, 0, 0)).all():
                        colored_mask[i, j] = (0, 0, 255)
                    elif (colored_mask[i, j] < (100, 100, 100)).all():
                        colored_mask[i, j] = (255, 0, 0)
                    else:
                        colored_mask[i, j] = (0, 255, 0)

        output = cv.bitwise_or(colored_image, colored_mask)

        output_path = 'out/'
        cv.imwrite((os.path.join(output_path, OUTPUT_NAME)), output)

        print(k)
        #cv.imshow("image", im)
        #cv.imshow("mask", ms)
        #cv.imshow(OUTPUT_NAME, output)

        #cv.waitKey(0)
        #cv.destroyAllWindows()


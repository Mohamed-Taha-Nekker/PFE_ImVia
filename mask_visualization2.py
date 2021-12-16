import cv2 as cv
import numpy as np
import os
import nibabel

# Dossier vers la base de données

images_path = 'ACDC_new/images/'
masks_path = 'ACDC_new/masks/'

# Organiser les images et les masques .nii en ordre

images = sorted(os.listdir(images_path))
masks = sorted(os.listdir(masks_path))

# Parcourir les elements de la base de donnée

for i, j in zip(images, masks):

    # Afficher l'image et le masque en cours de traitement
    print(i)
    print(j)

    n = str(i)

    image_filename = os.path.join(images_path, i)
    mask_filename = os.path.join(masks_path, j)

    # Lire les images et les masques .nii

    img = nibabel.load(image_filename)
    msk = nibabel.load(mask_filename)

    # Lire ces fichiers sous forme de matrices

    img = img.get_fdata()
    msk = msk.get_fdata()

    # Parcourir chaque slice des images et masques

    for k in range(img.shape[2]):

        # La Forme du nom de l'image de sortie

        OUTPUT_NAME = n + "_" + str(k) + "_output.jpg"

        im = np.float32(img[:,:,k])
        ms = np.float32(msk[:,:,k])

        # Transformation en images RGB

        colored_image = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
        colored_mask = cv.cvtColor(ms, cv.COLOR_GRAY2RGB)

        # Normalisation

        norm = np.zeros((800,800))

        colored_image = cv.normalize(colored_image, norm, 0, 255, cv.NORM_MINMAX)
        colored_mask = cv.normalize(colored_mask, norm, 0, 255, cv.NORM_MINMAX)

        # Changer la valeurs des pixels du masque de telle sorte que chaque partie soit d'une couleur spécifique

        for i in range(colored_mask.shape[0]):
            for j in range(colored_mask.shape[1]):
                if (colored_mask[i, j] > (80, 80, 80)).all():
                    if (colored_mask[i, j] > (230, 0, 0)).all():
                        colored_mask[i, j] = (0, 0, 255)
                    elif (colored_mask[i, j] < (100, 100, 100)).all():
                        colored_mask[i, j] = (255, 0, 0)
                    else:
                        colored_mask[i, j] = (0, 255, 0)

        # Ajuster la transparence du masque pour qu'il soit un peude couleur light

        colored_mask = cv.addWeighted(colored_mask, 0.4, colored_mask, 0.4, 0)

        # Crétion de l'image de sortie en fusionnant l'image original avec le masque colorisé

        output = cv.bitwise_or(colored_image, colored_mask)

        # Dossier de sortie

        output_path = 'out/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Stockage des images de sortie dans le dossier de sortie

        cv.imwrite((os.path.join(output_path, OUTPUT_NAME)), output)

        # Pour voir l'avancement du programme

        print(k)

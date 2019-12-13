from glob import glob
from skimage import io
from skimage import color
import numpy as np
import nipype as ni
import tensorlayer as tl
import os
import itk
import progressbar
import warnings
import matplotlib.pyplot as plt
from pre_process import Pre_process
from pre_process import Brats_dataset_generator

#Adapted from https://github.com/xf4j/brats17
#They achieve 0.9 Dice coefficient

def read_patches(path):
    image = np.load(path + '.npy')
    seg = image[..., -1]
    label = np.zeros((image.shape[0], image.shape[1], image.shape[2], 4), dtype=np.float32)
    label[seg == 0, 0] = 1
    label[seg == 1, 1] = 1
    label[seg == 2, 2] = 1
    label[seg == 4, 3] = 1
    return image[..., :-1], label


def get_patch_locations(patches_per_image, patch_size, image_size):
    nx = round((patches_per_image * 8 * image_size[2]* image_size[2] / image_size[1] / image_size[0]) ** (1.0 / 3))
    ny = round(nx * image_size[1] / image_size[2])
    nz = round(nx * image_size[0] / image_size[2])
    x = np.rint(np.linspace(patch_size, image_size[2] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, image_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, image_size[0] - patch_size, num=nz))

    return x, y, z


def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))

    return x, y, z


def generate_patch_probs(seg_image, patch_locations, patch_size, image_size):
    x, y, z = patch_locations
    seg = seg_image
    p = []
    for i in range(0, len(z)):
        for j in range(0, len(y)):
            for k  in range(0, len(x)):
                patch = seg[int(z[i] - patch_size/2):int(z[i] + patch_size/2),
                            int(y[j] - patch_size/2):int(y[j] + patch_size/2),
                            int(x[k] - patch_size/2):int(x[k] + patch_size/2)]
                # print(patch.shape)
                patch = (patch>0).astype(np.float32)
                #Foreground
                percent =  np.sum(patch)/(patch_size*patch_size*patch_size)
                # print(percent)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype = np.float32)
    p[p==0] = np.amin(p[np.nonzero(p)])
    p = p/np.sum(p)
    return p


def generate_patches(data_gen, patches_per_image=400, patch_size=32 , image_size=(155, 240, 240)):
    patches_per_image = patches_per_image
    patch_size = patch_size
    image_size = image_size
    #output_path = os.path.join(PATH, "patches")
    #print(output_path)
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)
    patch_locations = get_patch_locations(patches_per_image, patch_size, image_size)
    x, y, z = perturb_patch_locations(patch_locations, patch_size/16)
    # print(patch_locations)
    # print(x)
    # print(y)
    # print(z)
    patient_num = 0
    for img, gt in data_gen:
        image_concat = np.zeros((155, 240, 240, 5))
        image_concat[...,0:4] = img
        image_concat[...,4] = gt
        probs = generate_patch_probs(gt, (x, y, z), patch_size, image_size)
        selections = np.random.choice(range(len(probs)), size = patches_per_image, replace=False, p=probs)
        for num, sel in enumerate(selections):
            i, j, k = np.unravel_index(sel, (len(z), len(y), len(x)))
            patch = image_concat[int(z[i]-patch_size/2):int(z[i] + patch_size/2),
                                int(y[j] - patch_size/2):int(y[j] + patch_size/2),
                                int(x[k] - patch_size/2):int(x[k] +  patch_size/2),:]
            #np.save(output_path + ('/{}_{}.npy').format(patient_num, num), patch)
            yield patch[..., :4], patch[..., 4]
        patient_num += 1

def generate_patches_unbiased(data_gen, patches_per_image=400, patch_size=32 , image_size=(155, 240, 240)):
    patches_per_image = patches_per_image
    patch_size = patch_size
    image_size = image_size
    x,y, z = get_patch_locations(patches_per_image, patch_size, image_size)
    patient_num = 0
    for img, gt in data_gen:
        image_concat = np.zeros((155, 240, 240, 5))
        image_concat[...,0:4] = img
        image_concat[...,4] = gt
        probs = generate_patch_probs(gt, (x, y, z), patch_size, image_size)
        selections = np.random.choice(range(len(probs)), size = patches_per_image, replace=False)
        print(selections)
        for num, sel in enumerate(selections):
            i, j, k = np.unravel_index(sel, (len(z), len(y), len(x)))
            patch = image_concat[int(z[i]-patch_size/2):int(z[i] + patch_size/2),
                                int(y[j] - patch_size/2):int(y[j] + patch_size/2),
                                int(x[k] - patch_size/2):int(x[k] +  patch_size/2),:]
            #np.save(output_path + ('/{}_{}.npy').format(patient_num, num), patch)
            yield patch[..., :4], patch[..., 4]
        patient_num += 1


#if __name__ == '__main__':
#    generate_patches(Brats_dataset_generator(PATH), 400,32,image_size=(155, 240, 240))

from glob import glob
import numpy as np
import nipype as ni
import tensorlayer as tl
import os
import itk
import progressbar
import warnings
from data import to_tf_records_3D
import util
import generate_patches
import common

#To Do Bias Correction

class Pre_process(object):

    def __init__(self, path):
        self.path = path
        self.modals = ['Flair', 'T1', 'T1c', 'T2', 'OT']
        self.image, self.ground_truth = self.read_normalized()

    def read_normalized(self):

        flair_path = glob(self.path + '/*Flair*' + '/*Flair*')
        T1_path = glob(self.path + '/*T1.*' + '/*T1.*')
        T1c_path = glob(self.path + '/*T1c*' + '/*T1c*')
        T2_path = glob(self.path + '/*T2*' + '/*T2*')
        Gt_path = glob(self.path + '/*OT*' + '/*OT*')

        data_path = [flair_path[0], T1_path[0], T1c_path[0], T2_path[0], Gt_path[0]]

        slices = []

        for i in range(0, len(data_path)):
            scan_image = itk.imread(data_path[i])
            np_array = itk.GetArrayFromImage(scan_image)
            slices.append(np_array)


        image = np.stack(slices[:-1], axis=3)
        ground_truth = slices[-1]

        min_v, max_v = np.percentile(image, (0, 99.5))
        image = np.clip(image, min_v, max_v)

        image = image / np.max(image, axis=(0, 1, 2), keepdims=True)
        return image, ground_truth


def Brats_dataset_generator(HGG_path_list, LGG_path_list):
    # print(path)
    bar = progressbar.ProgressBar()
    # HGG_data_path = os.path.join(path, "HGG")
    # LGG_data_path = os.path.join(path, "LGG")

    # #Can be used to split train/val/test data
    # HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
    # LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)
    # print(len(HGG_path_list))
    # print(len(LGG_path_list))
    # print(dtype(HGG_path_list))
    # print(dtype(LGG_path_list))
    Data_list = [*HGG_path_list, *LGG_path_list]
    num_samples = len(Data_list)


    for patient_num in bar(range(len(Data_list))):
        path = Data_list[patient_num]
        pobject = Pre_process(path)
        # Uncomment this to visualize images
        #util.multi_slice_viewer(np.concatenate([pobject.image, pobject.ground_truth[..., np.newaxis]], axis=3))
        #util.to_gif(pobject.image[..., 0], "../data/example")
        yield pobject.image, pobject.ground_truth


def brats_to_tf_records(HGG_train_list, LGG_train_list, tf_record_path, patches_per_image, patch_size , image_size):
    image_gen = Brats_dataset_generator(HGG_train_list, LGG_train_list)
    patch_gen = generate_patches.generate_patches(
        image_gen,
        patches_per_image=patches_per_image,
        patch_size=patch_size,
        image_size=image_size
    )
    to_tf_records_3D(tf_record_path, patch_gen)

def brats_to_tf_records_val(HGG_val_list, LGG_val_list, tf_record_path, patches_per_image, patch_size, image_size):
    image_gen = Brats_dataset_generator(HGG_val_list, LGG_val_list)
    patch_gen = generate_patches.generate_patches(
        image_gen,
        patches_per_image=patches_per_image,
        patch_size=patch_size,
        image_size=image_size
    )
    to_tf_records_3D(tf_record_path, patch_gen)

def brats_to_tf_records_val_unbiased(HGG_val_list, LGG_val_list, tf_record_path, patches_per_image, patch_size, image_size):
    image_gen = Brats_dataset_generator(HGG_val_list, LGG_val_list)
    patch_gen_unbiased = generate_patches.generate_patches_unbiased(
        image_gen,
        patches_per_image=patches_per_image,
        patch_size=patch_size,
        image_size=image_size
    )
    to_tf_records_3D(tf_record_path, patch_gen_unbiased)

def brats_to_tf_records_test(HGG_test_list, LGG_test_list, tf_record_path):
    image_gen = Brats_dataset_generator(HGG_test_list,LGG_test_list)
    to_tf_records_3D(tf_record_path, image_gen)

def brats_to_tf_records_unbiased(HGG_train_list, LGG_train_list, tf_record_path, patches_per_image, patch_size, image_size):
    image_gen = Brats_dataset_generator(HGG_train_list, LGG_train_list)
    patch_gen_unbiased = generate_patches.generate_patches_unbiased(
        image_gen,
        patches_per_image=patches_per_image,
        patch_size=patch_size,
        image_size=image_size
    )
    to_tf_records_3D(tf_record_path, patch_gen_unbiased)

def generate_train_val_split(path, seed=0):
    np.random.seed(seed)
    HGG_data_path = os.path.join(path, "HGG")
    LGG_data_path = os.path.join(path, "LGG")

    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)

    # HGG -split 132/44/44
    # LGG -split 32/11/11
    # print(len(HGG_path_list))
    # print(len(LGG_path_list))
    HGG_train_list = np.random.choice(HGG_path_list, size = 132, replace = False)
    Remain_HGG_train_list = [item for item in HGG_path_list if item not in  HGG_train_list]
    HGG_val_list = np.random.choice(Remain_HGG_train_list, size = 44, replace = False)
    HGG_test_list = [item for item in Remain_HGG_train_list if item not in HGG_val_list]


    LGG_train_list = np.random.choice(LGG_path_list, size =  32, replace = False)
    Remain_LGG_train_list = [item for item in LGG_path_list if item not in  LGG_train_list]
    LGG_val_list = np.random.choice(Remain_LGG_train_list, size = 11, replace = False)
    LGG_test_list = [item for item in Remain_LGG_train_list if item not in LGG_val_list]

    return HGG_train_list, HGG_val_list,HGG_test_list, LGG_train_list, LGG_val_list, LGG_test_list




if __name__ == '__main__':
    path_to_dataset = common.INPUT_PATH
    path_to_tf_record = common.RECORD_PATH_TRAIN
    path_to_tf_record_unbiased = common.RECORD_PATH_TRAIN_UNBIASED
    path_to_tf_record_val = common.RECORD_PATH_VAL
    path_to_tf_record_test = common.RECORD_PATH_EVAL
    path_to_tf_record_val_unbiased = common.RECORD_PATH_VAL_UNBIASED
    # path_to_test_dataset = common.INPUT_PATH_TEST
    HGG_train_list, HGG_val_list, HGG_test_list, LGG_train_list, LGG_val_list, LGG_test_list = generate_train_val_split(path_to_dataset)
    #Generate tf record for training_data
    brats_to_tf_records(
        HGG_train_list,
        LGG_train_list,
        path_to_tf_record,
        patches_per_image=common.PATCHES_PER_IMAGE,
        patch_size=common.IMAGE_SIZE,
        image_size=common.ORIGINAL_IMAGE_SIZE
    )
    brats_to_tf_records_unbiased(
        HGG_train_list,
        LGG_train_list,
        path_to_tf_record_unbiased,
        patches_per_image=common.PATCHES_PER_IMAGE,
        patch_size=common.IMAGE_SIZE,
        image_size=common.ORIGINAL_IMAGE_SIZE
    )
    # #Generate tf record for validation_data
    brats_to_tf_records_val(
        HGG_val_list,
        LGG_val_list,
        path_to_tf_record_val,
        patches_per_image=common.PATCHES_PER_IMAGE,
        patch_size=common.IMAGE_SIZE,
        image_size=common.ORIGINAL_IMAGE_SIZE
    )
    brats_to_tf_records_val_unbiased(
        HGG_val_list,
        LGG_val_list,
        path_to_tf_record_val_unbiased,
        patches_per_image=common.PATCHES_PER_IMAGE,
        patch_size=common.IMAGE_SIZE,
        image_size=common.ORIGINAL_IMAGE_SIZE
    )
    brats_to_tf_records_test(HGG_test_list, LGG_test_list, path_to_tf_record_test)

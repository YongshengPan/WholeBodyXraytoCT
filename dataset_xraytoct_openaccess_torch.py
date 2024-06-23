import os
import time
import cv2
import numpy as np
import csv
from itertools import combinations
import random
import SimpleITK as sitk
from core.dataproc_utils import resize_image_itk, rotation3d, get_aug_crops, standard_normalization
from typing import Any, Callable, Dict, List, Optional, Tuple
from multiprocessing.pool import ThreadPool
import glob

class DataBase(object):
    datapool = {}

    def __init__(self,
                 input_path,
                 side_len=(128, 224, 224),
                 center_shift=(0, 16, 0),
                 data_shape=(128, 192, 224),
                 subset=('HNSCC', 'NSCLC', 'CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA'),
                 angle_of_views=(0, 45, 90, 135),
                 cycload=True,
                 aug_model='random',
                 use_augment=True,
                 aug_side=(256, 0, 0),
                 aug_stride=(32, 16, 16),
                 random_rotation=True,
                 random_views=None,
                 return_original=False,
                 randomcrop=(0, 1),
                 randomflip=('sk', 'flr', 'fud', 'r90')):
        self.side_len = side_len
        self.center_shift = center_shift
        self.data_shape = data_shape
        self.input_path = input_path
        self.cycload = cycload
        self.use_augment = use_augment
        self.aug_side = aug_side
        self.aug_stride = aug_stride
        self.aug_model = aug_model
        self.random_rotation = random_rotation
        self.random_views = random_views
        self.randomcrop = randomcrop
        self.randomflip = randomflip
        self.angle_of_views = angle_of_views
        self.return_original = return_original
        self.subset = subset
        self.cls_num = 2
        self.channels = {'CT': 1, 'skeleton': 5, 'ACPET': 1, 'NACPET': 1, 'bone': 1,  'Xray_views': len(angle_of_views), 'label': 2}
        self.dataset = 'PET2CT'
        # self.datapool = {}
        self.input_setup()

    def get_database(self, csvname, subset='train'):
        imdb = []
        with open(csvname, newline='') as csvfile:
            imdbreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if len(imdb) == 0:
                    pass
                elif os.path.join(self.input_path, row['Collection'], row['SubjectID'], row['Filepath']) == os.path.join(self.input_path, imdb[-1]['Collection'],  imdb[-1]['SubjectID'],  imdb[-1]['Filepath']):
                    row['SampleID'] = str(int(imdb[-1]['SampleID']) + 1)
                imdb.append(row)
                # if float(thinkness) < 4.0:
                # imdb.append([row['Collection'], row['Collection'], row['Collection'], row['Collection'], row['Collection'], row['Collection']])
        return imdb

    def input_setup(self):
        # subsitelist = ['CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'HNSCC', 'NSCLC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA']
        # trainsitelist = ['HNSCC', ]
        # validsitelist = ['CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA']
        # testsitelist = ['NSCLC', 'CPTAC-LSCC', 'CPTAC-LUAD', 'CPTAC-PDA', 'CPTAC-UCEC', 'TCGA-HNSC', 'TCGA-LUAD', 'TCGA-THCA']
        imdb = self.get_database(self.input_path + '/meta_all_collection.csv')
        # for item in imdb:
        #     in_cond = os.path.join(item['Collection'], item['SubjectID'], item['Filepath'])
            # scout = glob.glob(os.path.join(self.input_path, in_cond, '*SCOUT*.nii.gz'))
            # if len(scout) > 0:
            #     item['SCOUT'] = scout
            # print(item)

        # print(imdb)
        # imdb = pandas.read_csv(self.input_path + '/filelist.csv')
        self.imdb = [imitem for imitem in imdb if imitem['Collection'] in self.subset and len(imitem['CT']) > 6 and len(imitem['NACPET']) > 6]
        print(len(self.imdb))

    def read_images(self, item):
        """
        folders = "E:/dataset/NACdata1/HNSCC/HNSCC-01-0117/06-03-2000-NA-PETCT HEAD  NECK CA-68291"
        first = sitk.ReadImage(os.path.join(folders, "4.000000-PETAC-49490.nii.gz"))
        second = sitk.ReadImage(os.path.join(folders, "4.000000-PETAC-63662.nii.gz"))
        first_pad = sitk.ConstantPad(first, padLowerBound=[0, 0, 91], padUpperBound=[0, 0, 0])
        first_pad[:, :, 10:91] = second[:, :, 0:81]
        sitk.WriteImage(first_pad, os.path.join(folders, "4.000000-PETAC-49490E.nii.gz"))
        first = sitk.ReadImage(os.path.join(folders, "5.000000-PETNAC-16727.nii.gz"))
        second = sitk.ReadImage(os.path.join(folders, "5.000000-PETNAC-94706.nii.gz"))
        first_pad = sitk.ConstantPad(first, padLowerBound=[0, 0, 91], padUpperBound=[0, 0, 0])
        first_pad[:, :, 10:91] = second[:, :, 0:81]
        sitk.WriteImage(first_pad, os.path.join(folders, "5.000000-PETNAC-16727E.nii.gz"))
        """
        in_cond = os.path.join(item['Collection'], item['SubjectID'], item['Filepath'])
        subset = [item['SampleID'], item['ACPET'], item['NACPET'], item['CT'], item['SCOUT']]
        spacing = 2
        processed_path = os.path.join(self.input_path, in_cond, 'S%s' % (subset[0]))

        if not os.path.exists(processed_path):
            os.mkdir(processed_path)
        if True and not os.path.exists(os.path.join(processed_path, 'RS%s_CT_%dmm.nii.gz' % (subset[0], spacing))):
            nameFile_CT = os.path.join(self.input_path, in_cond, subset[3])
            CTIMG = sitk.Maximum(sitk.ReadImage(nameFile_CT, outputPixelType=sitk.sitkInt16) + 1000, 0)
            # CTIMG = sitk.SmoothingRecursiveGaussian(CTIMG, sigma=[1.0, 1.0, 1.0])
            BMask = sitk.SmoothingRecursiveGaussian(CTIMG, sigma=[1.0, 1.0, 1.0])
            BMask = sitk.GrayscaleFillhole(sitk.Cast(BMask >= 100, sitk.sitkFloat32)) + sitk.Cast(BMask >= 850, sitk.sitkFloat32) + \
                    sitk.Cast(BMask >= 990, sitk.sitkFloat32) + sitk.Cast(BMask >= 1150, sitk.sitkFloat32)
            CTIMG = resize_image_itk(CTIMG, newSpacing=[spacing, spacing, spacing], newSize=[512 // spacing, 512 // spacing, None])
            BMask = resize_image_itk(BMask, newSpacing=[spacing, spacing, spacing], newSize=[512 // spacing, 512 // spacing, None])

            sitk.WriteImage(CTIMG-1000, os.path.join(processed_path, 'RS%s_CT_%dmm.nii.gz' % (subset[0], spacing)))
            sitk.WriteImage(BMask, os.path.join(processed_path, 'RS%s_BM_%dmm.nii.gz' % (subset[0], spacing)))

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(CTIMG)
            resampler.SetDefaultPixelValue(0)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetInterpolator(sitk.sitkLinear)

            nameFile_NAC = os.path.join(self.input_path, in_cond, subset[2])
            NACIMG = sitk.Maximum(sitk.ReadImage(nameFile_NAC, outputPixelType=sitk.sitkFloat32), 0)
            NACIMG = resampler.Execute(NACIMG)
            NACIMG = standard_normalization(NACIMG, divide='mean', remove_tail=True)
            NACIMG = sitk.Cast(sitk.Minimum(sitk.Maximum(NACIMG * 1000.0, 0), 10000), sitk.sitkUInt16)
            sitk.WriteImage(NACIMG, os.path.join(processed_path, 'RS%s_NAC_%dmm.nii.gz' % (subset[0], spacing)))

            nameFile_AC = os.path.join(self.input_path, in_cond, subset[1])
            ACIMG = sitk.Maximum(sitk.ReadImage(nameFile_AC, outputPixelType=sitk.sitkFloat32), 0)
            ACIMG = resampler.Execute(ACIMG)
            ACIMG = standard_normalization(ACIMG, divide='mean', remove_tail=True)
            ACIMG = sitk.Cast(sitk.Minimum(sitk.Maximum(ACIMG * 1000.0, 0), 10000), sitk.sitkUInt16)
            sitk.WriteImage(ACIMG, os.path.join(processed_path, 'RS%s_AC_%dmm.nii.gz' % (subset[0], spacing)))

        else:
            CTIMG = sitk.ReadImage(os.path.join(processed_path, 'RS%s_CT_%dmm.nii.gz' % (subset[0], spacing)))+1000
            BMask = sitk.ReadImage(os.path.join(processed_path, 'RS%s_BM_%dmm.nii.gz' % (subset[0], spacing)))
            NACIMG = sitk.ReadImage(os.path.join(processed_path, 'RS%s_NAC_%dmm.nii.gz' % (subset[0], spacing)))
            ACIMG = sitk.ReadImage(os.path.join(processed_path, 'RS%s_AC_%dmm.nii.gz' % (subset[0], spacing)))
        # CTIMG = sitk.SmoothingRecursiveGaussian(CTIMG, sigma=[random.choice([0.5, 1.0, 1.5, 2.0])]*3)
        CTIMG = sitk.SmoothingRecursiveGaussian(CTIMG, sigma=[1.5,] * 3)
        # print('TotalSegmentator -i "' + os.path.join(in_cond, 'RS%s_CT_%dmm.nii.gz' % (subset[0], spacing)).replace('\\', '/') + '" -o "' + os.path.join(in_cond, 'RS%s_CT_%dmm_seg' % (subset[0], spacing)).replace('\\', '/') + '" --preview')
        scan_size = CTIMG.GetSize()
        padBound = [0 if scan_size[0] >= self.side_len[2] else (self.side_len[2]-scan_size[0]+1)//2,
                    0 if scan_size[1] >= self.side_len[1] else (self.side_len[1]-scan_size[1]+1)//2,
                    0 if scan_size[2] >= self.side_len[0] else (self.side_len[0]-scan_size[2]+1)//2]
        if np.sum(padBound) > 0:
            # print(padBound)
            CTIMG = sitk.ConstantPad(CTIMG, padLowerBound=[0, padBound[1], padBound[2]], padUpperBound=[0, padBound[1], padBound[2]])
            CTIMG = sitk.MirrorPad(CTIMG, padLowerBound=[padBound[0], 0, 0], padUpperBound=[padBound[0], 0, 0])
            NACIMG = sitk.ConstantPad(NACIMG, padLowerBound=[0, padBound[1], padBound[2]], padUpperBound=[0, padBound[1], padBound[2]])
            NACIMG = sitk.MirrorPad(NACIMG, padLowerBound=[padBound[0], 0, 0], padUpperBound=[padBound[0], 0, 0])
            ACIMG = sitk.ConstantPad(ACIMG, padLowerBound=[0, padBound[1], padBound[2]], padUpperBound=[0, padBound[1], padBound[2]])
            ACIMG = sitk.MirrorPad(ACIMG, padLowerBound=[padBound[0], 0, 0], padUpperBound=[padBound[0], 0, 0])

        def least_square_error(X, Y):
            P = (X > np.mean(X))
            k = (np.mean(Y*P)/np.mean(P) - np.mean(Y*(1-P))/np.mean(1-P)) / (np.mean(X*P)/np.mean(P) - np.mean(X*(1-P))/np.mean(1-P))
            b = np.mean(Y) - k*np.mean(X)
            return k, b

        nameFile_SCOUT = os.path.join(self.input_path, in_cond, subset[4])
        # if os.path.exists(nameFile_SCOUT) and not os.path.exists(os.path.join(processed_path, 'RS%s_SCOUT_%dmm.nii.gz' % (subset[0], spacing))):
        #
        #     view0 = sitk.Cast(sitk.SumProjection(CTIMG, projectionDimension=1) / 1000, sitk.sitkFloat32)
        #     sitk.WriteImage(view0, os.path.join(processed_path, 'RS%s_VIEW0_%dmm.nii.gz' % (subset[0], spacing)))
        #
        #     SCOUTIMG = sitk.ReadImage(nameFile_SCOUT, outputPixelType=sitk.sitkFloat32)
        #     SCOUTIMG.SetSpacing(SCOUTIMG.GetSpacing()[0:2]+(512.0,))
        #     resampler = sitk.ResampleImageFilter()
        #     resampler.SetReferenceImage(CTIMG)
        #     resampler.SetDefaultPixelValue(0)
        #     resampler.SetOutputPixelType(sitk.sitkFloat32)
        #     resampler.SetInterpolator(sitk.sitkLinear)
        #     SCOUTIMG = resampler.Execute(SCOUTIMG)
        #     SCOUTIMG = sitk.Cast(SCOUTIMG, sitk.sitkInt16)
        #     SCOUTIMG = sitk.Cast(sitk.SumProjection(SCOUTIMG, projectionDimension=1) / 1000, sitk.sitkFloat32)
        #     k, b = least_square_error(sitk.GetArrayFromImage(SCOUTIMG), sitk.GetArrayFromImage(view0))
        #     print(k, b)
        #     SCOUTIMG = SCOUTIMG*k + b
        #     sitk.WriteImage(SCOUTIMG, os.path.join(processed_path, 'RS%s_SCOUT_%dmm.nii.gz' % (subset[0], spacing)))
        #     print(np.mean(sitk.GetArrayFromImage(view0)), np.mean(sitk.GetArrayFromImage(SCOUTIMG)))
        # elif os.path.exists(os.path.join(processed_path, 'RS%s_SCOUT_%dmm.nii.gz' % (subset[0], spacing))):
        #     SCOUTIMG = sitk.ReadImage(os.path.join(processed_path, 'RS%s_SCOUT_%dmm.nii.gz' % (subset[0], spacing)))
        # else:
        #     SCOUTIMG = None
        SCOUTIMG = None

            # print(scan.GetSize(), scan.GetDimension())
        affine = {'spacing': CTIMG.GetSpacing(), 'origin': CTIMG.GetOrigin(), 'direction': CTIMG.GetDirection(), 'size': CTIMG.GetSize(),
                  'depth': CTIMG.GetDepth(), 'dimension': CTIMG.GetDimension()}
        # attenuation_factor = 0.184 / 10 * affine['spacing'][1]
        #
        # dataessamble = {'CT': CTIMG, 'Xray_views': Xray_views, 'affine': affine}
        dataessamble = {'CT': CTIMG, 'NAC': NACIMG, 'AC': ACIMG, 'SCOUT': SCOUTIMG, 'affine': affine}
        label = np.zeros(2, np.float32)
        label[random.randint(0, 1)] = 1
        dataessamble.update({'label': label})
        return dataessamble

    def __len__(self) -> int:
        return len(self.imdb)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = self.imdb[index]
        index_code = os.path.join(item['Collection'], item['SubjectID'], item['Filepath'], 'S%s' % (item['SampleID']))
        if index_code in self.datapool:
            dataessamble = self.datapool[index_code]
        else:
            if self.cycload:
                pool = ThreadPool(5)
                items = self.imdb[index:min(index+5, len(self.imdb))]
                dataessambles = pool.map(self.read_images, items)
                for it in range(len(items)):
                    index_code = os.path.join(items[it]['Collection'], items[it]['SubjectID'], items[it]['Filepath'], 'S%s' % (items[it]['SampleID']))
                    self.datapool[index_code] = dataessambles[it]
                dataessamble = self.datapool[index_code]
            else:
                dataessamble = self.read_images(item)

        aug_side = self.aug_side
        aug_step = np.maximum(self.aug_stride, 1)
        if not self.use_augment: self.aug_model = 'center'
        image_size = dataessamble['affine']['size'][::-1]
        aug_range = [min(aug_side[dim], (image_size[dim] - self.data_shape[dim] - self.center_shift[dim]) // 2) for dim in range(3)]
        aug_center = [(image_size[dim] + self.center_shift[dim] - self.data_shape[dim]) // 2 for dim in range(3)]
        aug_crops, count_of_augs = get_aug_crops(aug_center, aug_range, aug_step, aug_count=1, aug_index=(1,), aug_model=self.aug_model)
        aug_crops = [[sX1, sY1, sZ1, sX1 + self.data_shape[0], sY1 + self.data_shape[1], sZ1 + self.data_shape[2]] for sX1, sY1, sZ1 in aug_crops]
        sX = aug_crops[0][0], aug_crops[0][0] + self.data_shape[0]
        sY = aug_crops[0][1], aug_crops[0][1] + self.data_shape[1]
        sZ = aug_crops[0][2], aug_crops[0][2] + self.data_shape[2]
        # print(aug_crops)
        CTIMG, NACIMG, ACIMG, SCOUT = dataessamble['CT'], dataessamble['NAC'], dataessamble['AC'], dataessamble['SCOUT']
        Xray_views = []
        attenuation_factor = 0.184 / 10 * dataessamble['affine']['spacing'][1]
        if self.random_rotation:
            pre_angle = random.randint(-10, 10)
            CTIMG = rotation3d(CTIMG, [0, 0, pre_angle], False)
            NACIMG = rotation3d(NACIMG, [0, 0, pre_angle], False)
            ACIMG = rotation3d(ACIMG, [0, 0, pre_angle], False)
        else:
            pre_angle = 0
        if self.random_views is not None:
            angle_of_views = random.choice(self.random_views)
        else:
            angle_of_views = self.angle_of_views
        for angle in angle_of_views:
            if SCOUT is not None and angle == 0:
                view = sitk.Cast(SCOUT * attenuation_factor, sitk.sitkFloat32)
                view = sitk.Expand(view, (np.array(dataessamble['affine']['size']) // np.array(view.GetSize())).tolist())
                view = np.expand_dims(sitk.GetArrayFromImage(view), axis=0)
                Xray_views.append(view)
                continue
            rotscan = rotation3d(CTIMG, [0, 0, angle], False)
            view = sitk.Cast(sitk.SumProjection(rotscan, projectionDimension=1) / 1000 * attenuation_factor, sitk.sitkFloat32)
            view = sitk.Expand(view, (np.array(dataessamble['affine']['size']) // np.array(view.GetSize())).tolist())
            view = rotation3d(view, [0, 0, -angle], False)
            # cv2.imwrite(namepath + "/{0}_xray.tiff".format(angle),
            #             np.squeeze(np.uint8(np.minimum(np.maximum(sitk.GetArrayFromImage(view)*30, 0), 255))))
            view = np.expand_dims(sitk.GetArrayFromImage(view), axis=0)
            Xray_views.append(view)
        Xray_views = np.mean(Xray_views, axis=0)
        CTimg = np.expand_dims(np.float32(sitk.GetArrayFromImage(CTIMG)) * 0.001, axis=0)
        NACimg = np.expand_dims(np.float32(sitk.GetArrayFromImage(NACIMG)) * 0.001, axis=0)
        ACimg = np.expand_dims(np.float32(sitk.GetArrayFromImage(ACIMG)) * 0.001, axis=0)
        info = {'orig_size': image_size, 'aug_crops': aug_crops, 'affine': [dataessamble['affine']], 'flnm': [index_code], 'count_of_augs': [count_of_augs]}
        if self.return_original:
            datainput = {'CT': CTimg, 'NAC': NACimg, 'AC': ACimg, 'Xrays': Xray_views}
        else:
            datainput = {'CT': CTimg[:, sX[0]:sX[1], sY[0]:sY[1], sZ[0]:sZ[1]], 'Xrays': Xray_views[:, sX[0]:sX[1], sY[0]:sY[1], sZ[0]:sZ[1]],
                         'NAC': NACimg[:, sX[0]:sX[1], sY[0]:sY[1], sZ[0]:sZ[1]], 'AC': ACimg[:, sX[0]:sX[1], sY[0]:sY[1], sZ[0]:sZ[1]]}
        # datainput['SKT'] = np.concatenate((np.int8(datainput['CT'] <= 0.10), np.int8(datainput['CT'] > 0.10), np.int8(datainput['CT'] > 0.85), np.int8(
        #     datainput['CT'] > 0.99), np.int8(datainput['CT'] > 1.15)), axis=0)
        datainput['SKT'] = np.int8(datainput['CT'] > 0.10) + np.int8(datainput['CT'] > 0.85) + np.int8(datainput['CT'] > 0.99) + np.int8(datainput['CT'] > 1.15)
        info['label'] = dataessamble['label']
        return datainput, info

    def save_output(self, result_path, flnm, eval_out):
        flnm, refA, refB, synA, synB = eval_out['flnm'], eval_out['refA'], eval_out['refB'], eval_out['synA'], eval_out['synB']
        affine = eval_out['affine']
        if isinstance(flnm, bytes): flnm = flnm.decode()
        if not os.path.exists(result_path + "/{0}".format(flnm)): os.makedirs(result_path + "/{0}".format(flnm))
        # print(affine)
        for ref in ['refA', 'refB', 'synA', 'synB']:
            img = eval_out[ref]
            if img is not None:
                # img = np.pad(img, ((0, 0), (80, 80), (80, 80), (0, 0), ), )
                img = sitk.GetImageFromArray(np.int16(img * 1000)-1000)
                # img = sitk.GetImageFromArray(self.ct_rgb2gray(img))
                img.SetOrigin(affine['origin'])
                img.SetSpacing(affine['spacing'])
                img.SetDirection(affine['direction'])
                # img = sitk.ConstantPad(img, padLowerBound=[64, 64, 0], padUpperBound=[64, 64, 0], constant=-1000)
                sitk.WriteImage(img, result_path + "/{0}/{1}.nii.gz".format(flnm, ref), useCompression=True)
                evalout = eval_out[ref][::-1, ::-1, :]
                cv2.imwrite(result_path + "/{0}/{1}_coronal.tiff".format(flnm, ref), np.uint8(np.minimum(np.maximum(evalout[:, :, -128, 0] * 256 - 127, 0), 255)))
                cv2.imwrite(result_path + "/{0}/{1}_sagittal.tiff".format(flnm, ref), np.uint8(np.minimum(np.maximum(evalout[:, -112, :, 0] * 256 - 127, 0), 255)))
                cv2.imwrite(result_path + "/{0}/{1}_axial.tiff".format(flnm, ref), np.uint8(np.minimum(np.maximum(evalout[-256, :, :, 0] * 256 - 127, 0), 255)))
                if np.shape(evalout)[-1] > 1:
                    evalout = np.argmax(evalout, axis=-1)
                    cv2.imwrite(result_path + "/{0}/Seg_{1}_coronal.tiff".format(flnm, ref),
                                np.uint8(np.minimum(np.maximum(evalout[:, :, -128] * 64, 0), 255)))
                    cv2.imwrite(result_path + "/{0}/Seg_{1}_sagittal.tiff".format(flnm, ref),
                                np.uint8(np.minimum(np.maximum(evalout[:, -112, :] * 64, 0), 255)))
                    cv2.imwrite(result_path + "/{0}/Seg_{1}_axial.tiff".format(flnm, ref),
                                np.uint8(np.minimum(np.maximum(evalout[-256, :, :] * 64, 0), 255)))

    def read_output(self, result_path, imdb, index):
        flnm = os.path.join(imdb[index]['Collection'], imdb[index]['SubjectID'], imdb[index]['Filepath'],  'S%s' % (imdb[index]['SampleID']))
        eval_out = {'flnm': flnm, 'refA': None, 'refB': None, 'synA': None, 'synB': None}
        # print(affine)
        for ref in ['refA', 'refB', 'synA', 'synB']:
            img = sitk.ReadImage(result_path + "/{0}/{1}.nii.gz".format(flnm, ref))
            arr = sitk.GetArrayFromImage(img)
            eval_out[ref] = np.float32(arr/1000.0 + 1.0)

        return eval_out


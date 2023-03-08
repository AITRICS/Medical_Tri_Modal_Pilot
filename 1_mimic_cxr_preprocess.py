# run: nohup python preprocess_mimic_cxr.py > preprocess_cxr.log &
import os
import pydicom
import numpy as np

from PIL import Image
from monai import transforms
from monai.data import PILReader

# set directory
base_dir   = '/nfs/thena/MedicalAI/ImageData/public/MIMIC_CXR/data'
dcm_dir    = os.path.join(base_dir, 'files')
jpeg_dir   = os.path.join(base_dir, 'files_jpg')
resize_dir = '/mnt/aitrics_ext/ext01/alex/multi_modal/files_resize'

# search dcm files
for (dir_path, dirs, files) in os.walk(dcm_dir):
    for f in files:
        extension = os.path.splitext(f)[-1]
        if extension == '.dcm':
            # dicom, jpeg img path
            dcm_path  = os.path.join(dir_path, f)
            jpeg_path = os.path.join(dir_path.replace('files', 'files_jpg'), 
                                     f.replace('.dcm', '.jpg'))
            # resize jpeg img path
            resize_path  = jpeg_path.replace(jpeg_dir, resize_dir)
            os.makedirs('/'.join(resize_path.split('/')[:-1]), exist_ok=True)
            # check: if there is resized jpeg image already
            if os.path.exists(resize_path):
                print(f'[skip] resize path: {resize_path}')
                continue
            
            # check: if there is (converted) jpeg img 
            if os.path.exists(jpeg_path):
                try:
                    # load jpeg img -> PIL image
                    image = Image.open(jpeg_path)
                except:
                    print(f'[IO error] jpeg path: {jpeg_path}')
                    continue
            else:
                try:
                    # load dicom img
                    image = pydicom.read_file(dcm_path)
                    # save jpeg img from PIL image
                    image = Image.fromarray(image.pixel_array)
                    # image.save(jpeg_path) # cannot save new jpg: dir permission
                except:
                    print(f'[IO error] dcm path: {dcm_path}')
                    continue

            try: 
                # get resize shape - take shorter edge to 256
                w, h  = np.array(image).shape
                if w > h:
                    resize = (int(256*w/h), 256)
                elif w == h:
                    resize = (256, 256)
                else:
                    resize = (256, int(256*h/w))
            except:
                print(f'[size error] jpeg path: {jpeg_path}')
                continue

            try:
                # transform image - rescale (0-1), resize, centercrop, random rotation
                preprocess = transforms.Compose([
                    transforms.LoadImage(image_only=True, reader=PILReader()),
                    transforms.AddChannel(),
                    transforms.Resize(resize, mode='bilinear'),
                    transforms.CenterSpatialCrop((224,224)),
                ])
                resize_image = preprocess(jpeg_path)   # np array
                resize_image = Image.fromarray(resize_image[0,:,:].T)
            except:
                print(f'[transform error] dcm path: {dcm_path}')
                continue

            try: 
                # save resize img
                resize_image = resize_image.convert('L')   # for grayscale img
                resize_image.save(resize_path)
            except:
                print(f'[save error] resize path: {resize_path}')
                continue
            else:
                print(f'[save] resize path: {resize_path}')
        
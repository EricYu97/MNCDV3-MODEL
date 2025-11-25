import os
import glob
import rasterio
from PIL import Image
import numpy as np
def check_img_sizes(root_path):
    domains=os.listdir(root_path)
    domains_temporals=[os.listdir(os.path.join(root_path, domain)) for domain in domains]

    for temporals in domains_temporals:
        for name in temporals:
            if "." in name:
                temporals.remove(name)
    
    domains_temporals_dict=dict(zip(domains, domains_temporals))

    total_pixels=0

    for domain, temporal_list in domains_temporals_dict.items():
        # print(domain)
        shapes=[]
        for temporal in temporal_list:
            tiff_file=glob.glob(os.path.join(root_path, domain, temporal, '*.tif'))[0]
            # print(tiff_file)
            raster=rasterio.open(tiff_file)
            image=raster.read()
            
            shapes.append([image.shape[-2], image.shape[-1]])
            png_file=glob.glob(os.path.join(root_path, domain, temporal, '*.png'))[0]
            label=Image.open(png_file)
            # print(label.size)
            shapes.append([label.size[1], label.size[0]])
        print(domain,"c, h, w:", image.shape)
        check_shape=[ii for n,ii in enumerate(shapes) if ii not in shapes[:n]]
        # print(check_shape)
        total_pixels+=image.shape[-2]*image.shape[-1]
        if len(check_shape)==1:
            # print(f'{domain} PASSED.')
            pass
        else:
            print(f'{domain} NOT PASSED.')
    print("total_pixels", total_pixels)
    # print(domains_temporals_dict)
def convert_labels(root_path):

    pallet = np.array([[255, 255, 255],
                        [38, 70, 83],
                        [42, 157, 143],
                        [233, 196, 106],
                        [244, 162, 97],
                        [231, 111, 81]], np.uint8)
    def rgb2label(x):  
        mask_mapping = {
            (255, 255, 255): 0,
            (38, 70, 83): 1,
            (42, 157, 143): 2,
            (233, 196, 106): 3,
            (244, 162, 97): 4,
            (231, 111, 81): 5,
        }
        label=np.zeros(x.shape[:2], dtype=np.uint8)

        print(label.shape)
        for k in mask_mapping:
            label[(x == k).all(axis=2)] = mask_mapping[k]
        return label
    
    def label2rgb(x):
        mask_mapping = {
            0: (255, 255, 255),
            1: (38, 70, 83),
            2: (42, 157, 143),
            3: (233, 196, 106),
            4: (244, 162, 97),
            5: (231, 111, 81),
        }
        label=np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)

        print(label.shape)
        for k in mask_mapping:
            print(k)
            # label[x == k] = mask_mapping[k]
        return label
    
    domains=os.listdir(root_path)
    domains_temporals=[os.listdir(os.path.join(root_path, domain)) for domain in domains]

    for temporals in domains_temporals:
        for name in temporals:
            if "." in name:
                temporals.remove(name)
    
    domains_temporals_dict=dict(zip(domains, domains_temporals))

    for domain, temporal_list in domains_temporals_dict.items():
        # print(domain)
        for temporal in temporal_list:
            tiff_file=glob.glob(os.path.join(root_path, domain, temporal, '*.tif'))[0]
            # print(tiff_file)
            raster=rasterio.open(tiff_file)
            image=raster.read()
            # print("c, h, w:", image.shape)
            png_file=glob.glob(os.path.join(root_path, domain, temporal, '*.png'))[0]
            label=np.array(Image.open(png_file))
            # print(label.shape)
            label_np=rgb2label(label)
            print(np.unique(label_np))
            # print(label.size)

if __name__=="__main__":
    # check_img_sizes('/Users/yu34/Documents/MNCDV3/MineNetCDV3/Multi-Temporal_updated')
    check_img_sizes("MineNetCDV3_Processed_NEW")
    # convert_labels('/Users/weikangyu/Documents/MNCDV3/MineNetCDV3_Processed')
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as tfs

MNCD_Dataset_Patchwise_Stats={'mean':[0.1457, 0.1481, 0.1671, 0.1850, 0.2228, 0.2415, 0.2362, 0.2452, 0.1835],
                               'std':[0.0672, 0.0806, 0.1099, 0.1033, 0.0873, 0.0868, 0.0847, 0.1123, 0.1066],
                               'min':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               'max':[1.2238, 1.6399, 1.7502, 1.8151, 1.6399, 2.3133, 2.4306, 1.717, 2.6611]}

MNCD_Dataset_Original_Stats={'mean':[0.1431, 0.1453, 0.1640, 0.1821, 0.2204, 0.2393, 0.2342, 0.2431, 0.1814],
                              'std':[0.0670, 0.0801, 0.1092, 0.1029, 0.0874, 0.0873, 0.0852, 0.1134, 0.1070],
                              'min':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              'max':[1.2238, 1.6399, 1.7502, 1.8151, 1.6399, 2.3133, 2.4306, 1.717, 2.6611]}

class MNCDV3_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, normalization=None):
        self.root_path = root_path
        self.normalization = normalization
        self.get_domain_data_list(self.root_path)
        self.palette=np.array([[255, 255, 255],
                           [255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255],
                           [255, 255, 0],
                           [255, 0, 255]], np.uint8)
        self.normalize_function = tfs.Normalize(mean=MNCD_Dataset_Patchwise_Stats['mean'],std=MNCD_Dataset_Patchwise_Stats['std'])
    def get_domain_data_list(self, root_path):
        self.domains= os.listdir(root_path)
        self.all_data_list = []
        for domain in self.domains:
            domain_path = os.path.join(root_path, domain)
            # use pre+image to load patches names
            patches = os.listdir(os.path.join(domain_path, 'pre', 'image'))
            patches = [domain+'#'+patch.split('_')[1].split('.')[0] for patch in patches]
            self.all_data_list.extend(patches)

        return self.all_data_list
    
    def __len__(self):
        return len(self.all_data_list)
    
    @staticmethod
    def rgb_to_label(color_img: np.ndarray,
                 palette: np.ndarray,
                 unknown_label: int = -1) -> np.ndarray:
        """
        Convert an (H, W, 3) colorized label image to an (H, W) integer label image.

        Inputs
        - color_img: np.ndarray shape (H, W, 3), dtype uint8 (or convertible)
        - palette: np.ndarray shape (N, 3), dtype uint8 (each row is [R,G,B])
        - unknown_label: int to use where the color is not found in palette (default -1)

        Output
        - label_img: np.ndarray shape (H, W), dtype int32 with values in [0..N-1] or unknown_label
        """
        if color_img.ndim != 3 or color_img.shape[2] != 3:
            raise ValueError("color_img must be (H, W, 3)")

        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError("palette must be (N, 3)")

        # Ensure uint8 for bit packing
        img = color_img.astype(np.uint8)
        pal = palette.astype(np.uint8)

        # Pack RGB into a single integer key for fast comparisons: R<<16 | G<<8 | B
        keys_img = (img[..., 0].astype(np.uint32) << 16) | (img[..., 1].astype(np.uint32) << 8) | img[..., 2].astype(np.uint32)
        pal_keys = (pal[:, 0].astype(np.uint32) << 16) | (pal[:, 1].astype(np.uint32) << 8) | pal[:, 2].astype(np.uint32)

        # Map palette key -> label index
        mapping = {int(k): int(i) for i, k in enumerate(pal_keys)}

        # Initialize output with unknown label
        out = np.full(keys_img.shape, unknown_label, dtype=np.int32)

        # Set each palette color's pixels to its index
        # This loops over palette entries (usually small)
        for key, label_idx in mapping.items():
            out[keys_img == key] = label_idx

        return out
    
    def get_sample(self, sample):
        domain, sample_id = sample.split('#')
        sample_path = os.path.join(self.root_path, domain)
        # load pre image
        pre_image_path = os.path.join(sample_path, 'pre', 'image', f'pre_{sample_id}.npy')
        pre_image = torch.from_numpy(np.load(pre_image_path))
        # load post image
        post_image_path = os.path.join(sample_path, 'post', 'image', f'post_{sample_id}.npy')
        post_image = torch.from_numpy(np.load(post_image_path))
        # concatenate pre and post images along channel dimension

        if self.normalization:
            # S2 image normalization, recommended by ESA
            pre_image = pre_image/10000.0
            post_image = post_image/10000.0
            pre_image = self.normalize_function(pre_image)
            post_image = self.normalize_function(post_image)

        pre_label_path = os.path.join(sample_path, 'pre', 'label', f'pre_label_{sample_id}.png')
        pre_label = np.asarray(Image.open(pre_label_path))
        post_label_path = os.path.join(sample_path, 'post', 'label', f'post_label_{sample_id}.png')
        post_label = np.asarray(Image.open(post_label_path))

        pre_label = self.rgb_to_label(pre_label, self.palette)
        post_label = self.rgb_to_label(post_label, self.palette)

        return pre_image, post_image, pre_label, post_label
    
    def __getitem__(self, idx):
        sample = self.all_data_list[idx]
        pre_image, post_image, pre_label, post_label = self.get_sample(sample)
        change_label = (pre_label != post_label).astype(np.int32)

        pre_image, post_image = pre_image, post_image
        pre_label, post_label, change_label = pre_label, post_label, change_label

        return pre_image, post_image, pre_label, post_label, change_label

def main():
    root_path='MNCDV3_Bitemporal_Cropped_Size256_Step128_2788Samples'
    dataset = MNCDV3_Dataset(root_path, normalization=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for i, (pre_image, post_image, pre_label, post_label) in enumerate(dataloader):
        print(f'Batch {i}:')
        print(f'  Pre Image Shape: {pre_image}')
        print(f'  Post Image Shape: {post_image}')
        print(f'  Pre Label Shape: {pre_label.shape}')
        print(f'  Post Label Shape: {post_label.shape}')

        print(f'  Pre Label Unique Classes: {torch.unique(pre_label)}')
        print(f'  Post Label Unique Classes: {torch.unique(post_label)}')
        if i == 100:  # Just to limit output for demonstration
            break

if __name__=="__main__":
    main()
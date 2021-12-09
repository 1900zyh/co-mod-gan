import os 
import io 
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm 
from glob import glob 
from dnnlib import tflib
from training import misc
from zipfile import ZipFile 

def generate(checkpoint, image, mask, output, level, truncation):
    # load model 
    tflib.init_tf()
    _, _, Gs = misc.load_pkl(checkpoint)
    latent = np.random.randn(1, *Gs.input_shape[1:])
    output += f'_level0{level}'
    os.makedirs(output, exist_ok=True)

    # setup data list 
    img_list=sorted(glob(os.path.join(image, '*.png')))
    mask_zip = mask 
    mask_list = [f'pconv/{str(2000 * level + i).zfill(5)}.png' for i in range(2000)]
    mask_list = mask_list * int(len(img_list) / len(mask_list) + 1)
    # # set subset 
    # img_list = [img_list[i] for i in range(opts.split, len(img_list), opts.total)]
    # mask_list = [mask_list[i] for i in range(opts.split, len(mask_list), opts.total)]

    for img_path, mask_path in tqdm(zip(img_list, mask_list), total=len(img_list)): 
        real = np.asarray(Image.open(img_path)).transpose([2, 0, 1])
        real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
        mask = ZipFile(mask_zip).read(mask_path)
        mask = Image.open(io.BytesIO(mask)).convert('1')
        mask = np.asarray(mask, dtype=np.float32)[np.newaxis]
        # mask = np.asarray(Image.open(mask).convert('1'), dtype=np.float32)[np.newaxis]
        
        fake = Gs.run(latent, None, real[np.newaxis], mask[np.newaxis], truncation_psi=truncation)[0]
        fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
        fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
        fake = Image.fromarray(fake)
        fp = os.path.join(output, os.path.basename(img_path))
        fake.save(fp)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', default='co-mod-gan-places2-050000.pkl')
    parser.add_argument('-i', '--image', help='Original image path', default='../datasets/places2/center_crop_valid')
    parser.add_argument('-m', '--mask', help='Mask path', default='../datasets/pconv.zip')
    parser.add_argument('-o', '--output', help='Output (inpainted) image path', default='../places2_valid')
    parser.add_argument('-l', '--level', type=int, required=True)
    parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)

    args = parser.parse_args()
    generate(**vars(args))

if __name__ == "__main__":
    main()

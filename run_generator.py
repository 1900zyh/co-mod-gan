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

def generate(checkpoint, image, mask, output, truncation, level, split, total):
    # load model 
    tflib.init_tf()
    _, _, Gs = misc.load_pkl(checkpoint)
    latent = np.random.randn(1, *Gs.input_shape[1:])
    output += f'_level0{level}'
    os.makedirs(output, exist_ok=True)
    masked_dir = output + '_masked'
    os.makedirs(masked_dir, exist_ok=True)

    # setup data list 
    img_list=sorted(glob(os.path.join(image, '*.png')))
    mask_zip = mask 
    mask_list = [f'pconv/{str(2000 * level + i).zfill(5)}.png' for i in range(2000)]
    mask_list = mask_list * int(len(img_list) / len(mask_list) + 1)
    # # set subset 
    img_list = [img_list[i] for i in range(split, len(img_list), total)]
    mask_list = [mask_list[i] for i in range(split, len(mask_list), total)]

    for img_path, mask_path in tqdm(zip(img_list, mask_list), total=len(img_list)): 
        real = np.asarray(Image.open(img_path).convert('RGB')).transpose([2, 0, 1])
        mask = ZipFile(mask_zip).read(mask_path)
        mask = Image.open(io.BytesIO(mask)).convert('1')
        mask = np.asarray(mask, dtype=np.float32)[np.newaxis]

        fp = os.path.join(output, os.path.basename(img_path))
        real = real * (1 - mask)
        Image.fromarray(real.transpose([1,2,0]).astype(np.uint8)).save(os.path.join(masked_dir, os.path.basename(fp)))

        real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
        fake = Gs.run(latent, None, real[np.newaxis], 1.0 - mask[np.newaxis], truncation_psi=truncation)[0]
        fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
        fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
        fake = Image.fromarray(fake)
        fake.save(fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='co-mod-gan-places2-050000.pkl',
                        help='Network checkpoint path')
    parser.add_argument('-i', '--image', default='../datasets/places2/center_crop_valid',
                        help='Original image path', )
    parser.add_argument('-m', '--mask', default='../datasets/pconv.zip',
                        help='Mask path', )
    parser.add_argument('-o', '--output', default='../places2_valid',
                        help='Output (inpainted) image path', )
    parser.add_argument('-t', '--truncation', default=None,
            help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.')

    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--total', type=int, required=True)

    args = parser.parse_args()
    generate(**vars(args))

if __name__ == "__main__":
    main()

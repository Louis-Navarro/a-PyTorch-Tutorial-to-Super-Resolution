import torch
import matplotlib.pyplot as plt
# from imageio import imread, imwrite, get_reader, get_writer
from skvideo.io import FFmpegReader, FFmpegWriter
from skimage.io import imread, imsave
from skimage.transform import rescale
import numpy as np
from math import ceil, floor
from utils import convert_image
from tqdm import tqdm
import os
import argparse
import pathlib

torch.backends.cudnn.benchmark = True


class VideoReader(FFmpegReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.getShape()[0]

    def __iter__(self):
        for im in self.nextFrame():
            img = convert_image(im.copy(), 'pil', 'imagenet-norm').unsqueeze(0)
            yield img


class BatchedVideoReader(FFmpegReader):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = batch_size

    def __len__(self):
        return sum(divmod(self.getShape()[0], self.batch))

    def _len(self):
        return floor(self.getShape()[0] / self.batch)

    def __iter__(self):
        for _ in range(self._len()):
            out = [convert_image(next(self.nextFrame()), 'pil',
                                 'imagenet-norm').unsqueeze(0) for _ in range(self.batch)]
            out = torch.cat(out)
            yield out

        for im in self.nextFrame():
            img = convert_image(im.copy(), 'pil', 'imagenet-norm')
            yield img.unsqueeze(0)


def get_args():
    parser = argparse.ArgumentParser(
        description='A neural network to upscale images and video by 2 or 4')
    parser.add_argument('--model', '-m', required=True,
                        help='Relative or as_posix path to the model')
    parser.add_argument(
        '--output', '-o', help='Name of the file or folder to output the output', type=pathlib.Path)
    parser.add_argument(
        '--image', '-i', help='Relative or as_posix path to the image to upscale', type=pathlib.Path)
    parser.add_argument(
        '--video', '-v', help='Relative or as_posix path to the video to upscale', type=pathlib.Path)
    parser.add_argument(
        '--batch', '-b', help='Number of images from the video to compute at the same time', type=int, default=8)
    parser.add_argument(
        '--folder', '-f', help='Relative or as_posix path to a folder of images to upscale')
    parser.add_argument(
        '--scale', '-s', help='Factor of upscaling', type=int, choices=[2, 4], required=True)
    parser.add_argument(
        '--cpu', help='Force the usage of the cpu', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    if (args.image, args.folder, args.video).count(None) != 2:
        print('You have to specify exaclty one of --image, --video and --folder')
        exit()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.cpu else 'cpu')

    model = torch.load(args.model, map_location=device)['generator'].to(device)
    model.eval()

    with torch.no_grad():
        if args.image:
            img = imread(args.image.as_posix()).copy()

            out = model(convert_image(img, source='pil',
                        target='imagenet-norm').unsqueeze(0).to(device))
            out = out.squeeze(0).cpu().detach()
            out = convert_image(out, source='[-1, 1]', target='pil')
            out = np.array(out)

            file_name = args.output.as_posix()
            if not args.output.is_file():
                file_name = os.path.join(
                    file_name, args.image.name)
            if args.scale == 2:
                out = rescale(out, 0.5, multichannel=True,
                              anti_aliasing=True)
            imsave(file_name, out)

        elif args.folder:
            assert not args.output.is_file()
            for fp in os.listdir(args.folder.as_posix()):
                img = imread(os.path.join(args.folder.as_posix(), fp)).copy()

                out = model(convert_image(img, source='pil',
                            target='imagenet-norm').unsqueeze(0).to(device))
                out = out.squeeze(0).cpu().detach()
                out = convert_image(out, source='[-1, 1]', target='pil')
                out = np.array(out)

                file_name = os.path.join(args.output.as_posix(), fp)
                if args.scale == 2:
                    out = rescale(out, 0.5)
                imsave(file_name, out)

        else:  # Video
            file_name = args.output.as_posix()
            if not args.output.is_file():
                file_name = os.path.join(
                    file_name, args.video.name)

            with FFmpegWriter(file_name) as out_file, BatchedVideoReader(args.batch, args.video.as_posix()) as imgs:
                for img in tqdm(imgs):
                    img = img.to(device)

                    img = model(img).cpu().detach()
                    for im in img:
                        im = convert_image(im, source='[-1, 1]', target='pil')
                        im = np.array(img)
                        if args.scale == 2:
                            im = rescale(im, 0.5)
                        out_file.writeFrame(im)


if __name__ == '__main__':
    args = get_args()
    main(args)

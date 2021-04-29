import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import adjust_learning_rate, AverageMeter, convert_image, clip_gradient
import argparse

"""
# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
# kernel size of the first and last convolutions which transform the inputs and outputs
large_kernel_size_g = 9
# kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
small_kernel_size_g = 3
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
# filepath of the trained SRResNet checkpoint used for initialization
srresnet_checkpoint = "checkpoints/checkpoint_srresnet.pth.tar"

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
"""
grad_clip = None  # clip if gradients are exploding


def get_args():
    parser = argparse.ArgumentParser(
        description='Module to train the SRGAN model')

    data_params = parser.add_argument_group('Data parameters')
    gen_params = parser.add_argument_group('Generator parameters')
    disc_params = parser.add_argument_group('Discriminator parameters')
    learn_params = parser.add_argument_group('Learning parameters')

    data_params.add_argument(
        '--data', '-d', help='Path to folder with JSON data files', default='./')
    data_params.add_argument(
        '--crop', help='Crop size of target HR images', type=int, default=96)
    data_params.add_argument(
        '--scaling', help='The scaling factor of the generator', type=int, default=4)

    gen_params.add_argument(
        '--large-kernel', help='Size of the large kernel in the generator', dest='large_kernel', type=int, default=9)
    gen_params.add_argument(
        '--small-kernel', help='Size of the small kernel in the generator', dest='small_kernel', type=int, default=3)
    gen_params.add_argument(
        '--channels-gen', help='Number of output channels inbetween input/output', dest='channels_g', type=int, default=64)
    gen_params.add_argument(
        '--blocks-gen', help='Number of residual blocks', dest='blocks_g', type=int, default=16)
    gen_params.add_argument(
        '--srresnet', '-s', help='Path to the trained srresnet checkpoint', required=True)

    disc_params.add_argument(
        '--kernel', help='Size of the kernel in the discriminator', dest='kernel', type=int, default=3)
    disc_params.add_argument(
        '--channels-disc', help='Number of output channels in the first conv block', dest='channels_d', type=int, default=64)
    disc_params.add_argument(
        '--blocks-disc', help='Number of residual blocks', dest='blocks_d', type=int, default=8)
    disc_params.add_argument(
        '--fc', help='Size of the first fully connected layer', type=int, default=1024)

    learn_params.add_argument('--checkpoint', '-cpt',
                              help='Path to SRGAN checkpoint')
    learn_params.add_argument(
        '--batch', help='Batch size', type=int, default=16)
    learn_params.add_argument(
        '--epoch', help='Start at this epoch', type=int, default=0)
    learn_params.add_argument(
        '--iterations', help='Number of training iterations', type=int, default=int(2e5))
    learn_params.add_argument(
        '--workers', help='Number of workers in DataLoader', type=int, default=4)
    learn_params.add_argument(
        '--vggi', help='The index i in the definition for VGG loss', type=int, default=5)
    learn_params.add_argument(
        '--vggj', help='The index j in the definition for VGG loss', type=int, default=4)
    learn_params.add_argument(
        '--beta', help='coefficient to weight the adversarial loss in the perceptual loss', type=float, default=1e-3)
    learn_params.add_argument(
        '--print', '--freq', help='Print training status every __ batches', type=int, default=500)
    learn_params.add_argument(
        '--lr', help='Learning rate', type=float, default=1e-4)

    parser.add_argument('--cpu', action='store_true',
                        help='Force the usage of the cpu')

    return parser.parse_args()


def main():
    """
    Training.
    """
    start_epoch = args.epoch
    checkpoint = args.checkpoint

    # Default device
    cudnn.benchmark = True

    # Initialize model or load checkpoint
    if args.checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size=args.large_kernel,
                              small_kernel_size=args.small_kernel,
                              n_channels=args.channels_g,
                              n_blocks=args.blocks_g,
                              scaling_factor=args.scaling)

        # Initialize generator network with pretrained SRResNet
        generator.initialize_with_srresnet(
            srresnet_checkpoint=args.srresnet, map_location=device)

        # Initialize generator's optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=args.lr)

        # Discriminator
        discriminator = Discriminator(kernel_size=args.kernel,
                                      n_channels=args.channels_d,
                                      n_blocks=args.blocks_d,
                                      fc_size=args.fc)

        # Initialize discriminator's optimizer
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        print("\nLoaded checkpoint from epoch %d.\n" %
              (checkpoint['epoch'] + 1))

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=args.vggi, j=args.vggj)
    truncated_vgg19.eval()

    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(args.data,
                              split='train',
                              crop_size=args.crop,
                              scaling_factor=args.scaling,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                               pin_memory=True)

    # Total number of epochs to train for
    epochs = int(args.iterations // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((args.iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch's training
        train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              truncated_vgg19=truncated_vgg19,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                   f'checkpoint_srgan_{epoch}.pth.tar')


def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        # (batch_size (N), 3, 24, 24), imagenet-normed
        lr_imgs = lr_imgs.to(device)
        # (batch_size (N), 3, 96, 96), imagenet-normed
        hr_imgs = hr_imgs.to(device)

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        # (N, 3, 96, 96), imagenet-normed
        sr_imgs = convert_image(
            sr_imgs, source='[-1, 1]', target='imagenet-norm')

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        # detached because they're constant, targets
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(
            sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(
            sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + args.beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
            adversarial_loss_criterion(
                hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))

    # free some memory since their histories may be stored
    del (lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space,
         sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated)


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    main()

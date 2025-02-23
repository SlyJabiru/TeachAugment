from . import augmentation_container
from . import cutout
from . import imagenet_augmentation
from . import nn_aug
from . import replay_buffer


def get_transforms(dataset):
    import torchvision.transforms as T
    if dataset == 'ImageNet':
        normalizer = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_transform = T.Compose([imagenet_augmentation.EfficientNetRandomCrop(224),
                                  T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                                  T.ToTensor()])
        base_aug = [T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    imagenet_augmentation.Lighting(),
                    normalizer]
        val_transform = T.Compose([imagenet_augmentation.EfficientNetCenterCrop(224),
                                   T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                                   T.ToTensor(),
                                   normalizer])
    elif dataset == 'MNIST':
        normalizer = T.Normalize((0.1307,), (0.3081,))
        train_transform = T.Compose([T.RandomCrop(28, padding=4),
                                     T.ToTensor()])
        base_aug = [normalizer,
                    cutout.Cutout(size=14)]
        val_transform = T.Compose([T.ToTensor(), normalizer])
    else:
        normalizer = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor()])
        base_aug = [normalizer,
                    cutout.Cutout()]
        val_transform = T.Compose([T.ToTensor(), normalizer])
    return base_aug, train_transform, val_transform, normalizer


def build_augmentation(n_classes, n_channel,
                       g_offset, g_scale, g_scale_unlimited,
                       c_scale, c_scale_unlimited, c_shift_unlimited,
                       c_reg_coef=0, normalizer=None, replay_buffer=None,
                       n_chunk=16, with_context=True):
    g_aug = nn_aug.GeometricAugmentation(n_classes, g_offset, g_scale, g_scale_unlimited, with_context=with_context)
    c_aug = nn_aug.ColorAugmentation(n_classes, n_channel, c_scale, c_scale_unlimited, c_shift_unlimited, with_context=with_context)
    augmentation = augmentation_container.AugmentationContainer(c_aug, g_aug, c_reg_coef, normalizer, replay_buffer, n_chunk)
    return augmentation

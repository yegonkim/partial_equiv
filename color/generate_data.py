import os
import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
# import torchvision.transforms.functional as F
import pandas as pd
import scipy

from omegaconf import OmegaConf

from color.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split

def generate_stl10(cfg: OmegaConf):
    # ImageNet-style preprocessing.
    tr_train = transforms.Compose(
        [
            # transforms.ColorJitter(
            #     brightness=0,
            #     contrast=0,
            #     saturation=0,
            #     hue=0,
            # ),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    tr_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    
    x_train = torchvision.datasets.STL10(
        'data', split="train", transform=tr_train, download=True
    )
    x_test = torchvision.datasets.STL10(
        'data', split="test", transform=tr_test, download=True
    )

    return x_train, x_test

def generate_cifar10(cfg: OmegaConf):
    dataset = CIFAR10
    validation_split = [45000, 5000]

    training_set = dataset(
        partition="train",
        augment=cfg.augment,
        rot_interval=cfg.dataset_params.rot_interval,
    )
    print("training set size", len(training_set))
    test_set = dataset(
        partition="test",
        augment="None",
        rot_interval=cfg.dataset_params.rot_interval,
    )

    training_set, validation_set = random_split(
        training_set,
        validation_split,
        generator=torch.Generator().manual_seed(42),
    )

    return training_set, validation_set, test_set

### Long-tailed ColorMNIST
# region
def generate_set(dataset, samples_per_class, train, bg_noise_std=0.1, bg_intensity=0.33) -> TensorDataset:
    """Generate 30-class color mnist dataset with long-tailed distribution."""

    imgs, targets = dataset.data.numpy(), dataset.targets.numpy()

    if train:
        # Create power law distribution for 30 classes.
        samples = np.random.power(0.3, size=imgs.shape[0]) * samples_per_class
        samples = np.ceil(samples).astype(int)
    else:
        # Create uniform distribution for 30 classes of 250 samples each.
        samples_per_class = 250
        samples = (np.ones(imgs.shape[0]) * samples_per_class).astype(int)

    # Convert to 30 classes with 3 colors per digit.
    imgs_rgb = []
    targets_rgb = []
    for i in range(10):
        samples_added = 0
        for j in range(3):
            class_idx = i * 3 + j

            # Get data.
            data_tmp = imgs[targets == i][
                samples_added : samples_added + samples[class_idx]
            ]
            # Create 3 channels and add data to j-th channel.
            data = np.zeros(data_tmp.shape + (3,))
            data[:, :, :, j] = data_tmp

            # Add data to list.
            imgs_rgb.append(data)
            targets_rgb.extend(list(np.ones(data.shape[0]) * class_idx))
            samples_added += samples[i]

    # Concatenate samples and targets.
    imgs_rgb = np.concatenate(imgs_rgb) / 255
    targets_rgb = np.asarray(targets_rgb)

    # Generate noisy background.
    ims = imgs_rgb.shape[:3]
    weight = np.max(imgs_rgb, axis=3)
    noisy_background = bg_intensity + np.random.randn(*ims) * bg_noise_std
    noisy_background = np.clip(noisy_background, 0, 1)
    # Add background to images.
    imgs_rgb = (
        weight[..., None] * imgs_rgb
        + (1 - weight[..., None]) * noisy_background[..., None]
    )

    # Convert to tensor.
    imgs_rgb = torch.from_numpy(imgs_rgb).permute(0, 3, 1, 2).float()
    targets = torch.from_numpy(targets_rgb).long()

    return TensorDataset(imgs_rgb, targets)

def generate_colormnist_longtailed(samples_per_class=150, datapath='data', bg_noise_std=0.1, bg_intensity=0.33):
    trainset = MNIST(
        root= datapath + "/MNIST",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    testset = MNIST(
        root= datapath + "/MNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    trainset = generate_set(trainset, samples_per_class, True, bg_noise_std, bg_intensity)
    # trainset_gray = TensorDataset(
    #     trainset.tensors[0].mean(1, keepdim=True), trainset.tensors[1]
    # )
    testset = generate_set(testset, samples_per_class, False, bg_noise_std, bg_intensity)

    return trainset, testset
# endregion

def convert_images_to_tensors(folder_path, extension='.jpg'):
    tensor_list = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith(extension)])
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array, dtype=torch.float).permute(2, 0, 1) / 255.0  # Normalize to floats between 0 and 1, color, x, y
        tensor_list.append(image_tensor)
    return tensor_list # list of tensors

def generate_gtsrb_data(size=224, dpath="./data/GTSRB"):
    if not os.path.exists(dpath+f'/train.pt'):
        images_list = []
        labels_list = []
        for class_folder in tqdm(os.listdir(dpath+'/GTSRB/Final_Training/Images')):
            try:
                label = int(class_folder)
            except:
                continue
            images = convert_images_to_tensors(dpath+'/GTSRB/Final_Training/Images/'+class_folder, extension='.ppm')
            images_list.extend(images)
            labels = [label] * len(images)
            labels_list.extend(labels)
        # for i in tqdm(range(len(images_list))):
        #     images_list[i] = torch.nn.functional.interpolate(images_list[i].unsqueeze(0), size=size).squeeze(0)
        torch.save([images_list, labels_list], dpath+f'/train.pt')
    else:
        images_list, labels_list = torch.load(dpath+f'/train.pt')
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, images_list, labels_list):
            self.images_list = images_list
            self.labels_list = labels_list
            self.transform = transforms.Resize(size=(size,size), interpolation=transforms.InterpolationMode.NEAREST)

        def __len__(self):
            return len(self.images_list)

        def __getitem__(self, index):
            image = self.images_list[index]
            image = self.transform(image)
            label = self.labels_list[index]
            return image, label

    traindata = CustomDataset(images_list, labels_list)

    if not os.path.exists(dpath+f'/test.pt'):
        images_list = convert_images_to_tensors(dpath+'/GTSRB/Final_Test/Images', extension='.ppm')
        # for i in tqdm(range(len(images_list))):
        #     images_list[i] = torch.nn.functional.interpolate(images_list[i].unsqueeze(0), size=size).squeeze(0)
        torch.save(images_list, dpath+f'/test.pt')
    else:
        images_list = torch.load(dpath+f'/test.pt')
    
    rotated_images_list = []
    rotation_transform = transforms.RandomRotation(degrees=(-10, 10), interpolation=transforms.InterpolationMode.BILINEAR)

    for image in images_list:
        rotated_image = rotation_transform(image)
        rotated_images_list.append(rotated_image)

    df = pd.read_csv(dpath+'/GT-final_test.csv', sep=';')
    labels_list = df.iloc[:, -1].tolist()
    testdata = CustomDataset(images_list, labels_list)

    dataset = {'train': traindata, 'test': testdata}
    return dataset # dictionary of datasets

def generate_102flower_data(size=224, root='data'):
    tr_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    tr_test = transforms.Compose(
        [   
            transforms.Resize(256),
            transforms.CenterCrop(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )

    x_train = torchvision.datasets.Flowers102(
        root, split="train", transform=tr_train, download=True
    )
    x_val = torchvision.datasets.Flowers102(
        root, split="val", transform=tr_train, download=True
    )
    x_train = torch.utils.data.ConcatDataset([x_train, x_val])  # type: ignore
    x_test = torchvision.datasets.Flowers102(
        root, split="test", transform=tr_test, download=True
    )

    dataset = {'train': x_train, 'test': x_test}
    return dataset # dictionary of datasets

    # # dpath contains datasplits.mat and jpg folder of 17flowers dataset
    # data_split = scipy.io.loadmat(dpath+'/setid.mat')
    # labels = scipy.io.loadmat(dpath+'/imagelabels.mat')['labels'][0]

    # if not os.path.exists(dpath+f'/images_{size}.pt'):
    #     images = convert_images_to_tensors(dpath+'/jpg')
    #     images= [transforms.functional.center_crop(image, [500, 500]) for image in tqdm(images)]
    #     for i in tqdm(range(len(images))):
    #         images[i] = torch.nn.functional.interpolate(images[i].unsqueeze(0), size=size).squeeze(0)
    #     torch.save(images, dpath+f'/images_{size}.pt')
    # else:
    #     images = torch.load(dpath+f'/images_{size}.pt')

    # train_images = []
    # train_labs = []

    # trn_idx = data_split['trnid'][0].squeeze()-1
    # val_idx = data_split['valid'][0].squeeze()-1
    # tst_idx = data_split['tstid'][0].squeeze()-1

    # train_images = torch.stack([images[i] for i in trn_idx])
    # train_labs = torch.tensor([labels[i]-1 for i in trn_idx])
    # val_images = torch.stack([images[i] for i in val_idx])
    # val_labs = torch.tensor([labels[i]-1 for i in val_idx])
    # test_images = torch.stack([images[i] for i in tst_idx])
    # test_labs = torch.tensor([labels[i]-1 for i in tst_idx])

    # traindata = torch.utils.data.TensorDataset(train_images, train_labs)
    # valdata = torch.utils.data.TensorDataset(val_images, val_labs)
    # testdata = torch.utils.data.TensorDataset(test_images, test_labs)
    # dataset = {'train': traindata, 'val': valdata, 'test': testdata}

    # return dataset # dictionary of datasets



def shaded_rectangle(coord, color):
    # coord: bs x 4, [x1, x2, y1, y2]
    # color: bs x 3
    bs = coord.shape[0]
    assert color.shape[0] == bs
    images = torch.zeros(bs, 3, 28, 28)
    for i in range(bs):
        images[i, :, coord[i,0]:(coord[i,0]+coord[i,1])//2, coord[i,2]:coord[i,3]] = color[i, :,None,None]
        images[i, :, (coord[i,0]+coord[i,1])//2:coord[i,1], coord[i,2]:coord[i,3]] = 1

    return images

def generate_shaded_rectangle_data_(ntrain_each, draw_bar = False, dataseed=88):
    # interpolation = transforms.InterpolationMode.BILINEAR
    interpolation = transforms.InterpolationMode.NEAREST

    torch.manual_seed(dataseed)
    images = []
    margin = 7
    lbound = margin
    ubound = 28-margin
    side = ubound-lbound
#######
    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(6, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)*0.9
    squares = shaded_rectangle(coord, color) # ntrain_each x 3 x 28 x 28

    angles = torch.rand(ntrain_each)*180-90
    rotated_images = []
    for img, angle in zip(squares, angles):
        rotated_img = F.rotate(img, angle.item(), interpolation=interpolation)
        rotated_images.append(rotated_img)
    squares = torch.stack(rotated_images)
    if draw_bar:
        squares[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares)
######
    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(6, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)*0.9
    squares_rot = shaded_rectangle(coord, color) # ntrain_each x 3 x 28 x 28

    angles = torch.rand(ntrain_each)*180+90
    rotated_images = []
    for img, angle in zip(squares_rot, angles):
        rotated_img = F.rotate(img, angle.item(), interpolation=interpolation)
        rotated_images.append(rotated_img)
    squares_rot = torch.stack(rotated_images)
    if draw_bar:
        squares_rot[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares_rot)
#########
    images = torch.cat(images, dim=0) # 4*ntrain_each x 3 x 28 x 28
    
    labels = [0,1]
    labels = [torch.ones(ntrain_each)*label for label in labels]
    labels = torch.cat(labels, dim=0).type(torch.long)

    return images, labels

def generate_shaded_rectangle_data(ntrain_each=1000, ntest_each=500, draw_bar = False, dataseed=88):
    torch.manual_seed(dataseed)
    train_images, train_labels = generate_shaded_rectangle_data_(ntrain_each, draw_bar, dataseed)
    test_images, test_labels = generate_shaded_rectangle_data_(ntest_each,  draw_bar, dataseed+1)
    
    traindata = torch.utils.data.TensorDataset(train_images, train_labels)
    testdata = torch.utils.data.TensorDataset(test_images, test_labels)

    return traindata, testdata


############

def generate_shape_data_binary_(ntrain_each, draw_bar = False, dataseed=88):
    # interpolation = transforms.InterpolationMode.BILINEAR
    interpolation = transforms.InterpolationMode.NEAREST

    torch.manual_seed(dataseed)
    images = []
    margin = 7
    lbound = margin
    ubound = 28-margin
    side = ubound-lbound
#######
    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(6, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)*0.9 + 0.1
    squares = rectangle(coord, color) # ntrain_each x 3 x 28 x 28

    squares = F.rotate(squares, 45)
    squares = F.rotate(squares, 45)

    angles = (torch.randint(0,8,(ntrain_each,))*2) * 45 # 짝수
    rotated_images = []
    for img, angle in zip(squares, angles):
        rotated_img = F.rotate(img, angle.item(), interpolation=interpolation)
        rotated_images.append(rotated_img)
    squares = torch.stack(rotated_images)
    if draw_bar:
        squares[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares)
######
    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(6, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)*0.9 + 0.1
    squares_rot = rectangle(coord, color) # ntrain_each x 3 x 28 x 28

    squares_rot = F.rotate(squares_rot, 45)
    squares_rot = F.rotate(squares_rot, 45)

    angles = (torch.randint(0,8,(ntrain_each,))*2+1) * 45 # 홀수
    rotated_images = []
    for img, angle in zip(squares_rot, angles):
        rotated_img = F.rotate(img, angle.item(), interpolation=interpolation)
        rotated_images.append(rotated_img)
    squares_rot = torch.stack(rotated_images)
    if draw_bar:
        squares_rot[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares_rot)
#########
    images = torch.cat(images, dim=0) # 4*ntrain_each x 3 x 28 x 28

    # # add fourth channel with zeros
    # images = torch.cat((images, torch.zeros(images.shape[0], 1, 28, 28)), dim=1)
    
    labels = [0,1]
    labels = [torch.ones(ntrain_each)*label for label in labels]
    labels = torch.cat(labels, dim=0).type(torch.long)

    return images, labels

def generate_shape_data_binary(ntrain_each=1000, ntest_each=500, draw_bar = False, dataseed=88):
    torch.manual_seed(dataseed)
    train_images, train_labels = generate_shape_data_binary_(ntrain_each, draw_bar, dataseed)
    test_images, test_labels = generate_shape_data_binary_(ntest_each,  draw_bar, dataseed+1)
    
    traindata = torch.utils.data.TensorDataset(train_images, train_labels)
    testdata = torch.utils.data.TensorDataset(test_images, test_labels)

    return traindata, testdata





def rectangle(coord, color):
    # coord: bs x 4, [x1, x2, y1, y2]
    # color: bs x 3
    bs = coord.shape[0]
    assert color.shape[0] == bs
    images = torch.zeros(bs, 3, 28, 28)
    for i in range(bs):
        images[i, :, coord[i,0]:coord[i,1], coord[i,2]:coord[i,3]] = color[i, :,None,None]

    return images

def generate_shape_data_(ntrain_each, draw_bar = False, dataseed=88):
    torch.manual_seed(dataseed)
    images = []
    margin = 7
    lbound = margin
    ubound = 28-margin
    side = ubound-lbound

    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(3, side, (ntrain_each, 2)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1)
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        if coord[i,1] - coord[i,0] == coord[i,3] - coord[i,2]:
            coord[i,3] += 1
    color = torch.rand(ntrain_each, 3)
    rectangles = rectangle(coord, color) # ntrain_each x 3 x 28 x 28
    if draw_bar:
        rectangles[:, :, -3:-1, 8:-8]= 1.0
    images.append(rectangles)

    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(3, side, (ntrain_each, 2)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1)
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        if coord[i,1] - coord[i,0] == coord[i,3] - coord[i,2]:
            coord[i,3] += 1
    color = torch.rand(ntrain_each, 3)
    rectangles_rot = rectangle(coord, color) # ntrain_each x 3 x 28 x 28
    if draw_bar:
        rectangles_rot[:, :, -3:-1, 8:-8]= 1.0
    rectangles_rot = F.rotate(rectangles_rot, 45)
    images.append(rectangles_rot)

    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(3, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)
    squares = rectangle(coord, color) # ntrain_each x 3 x 28 x 28
    if draw_bar:
        squares[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares)

    coord_1 = torch.randint(lbound+3, ubound-3, (ntrain_each, 2)) # x1, y1
    coord_2 = coord_1 + torch.randint(3, side, (ntrain_each, 1)) # x2, y2
    coord = torch.stack((coord_1[:,0], coord_2[:,0], coord_1[:,1], coord_2[:,1]), dim=1) # ntrain_each x 4
    coord = torch.clamp(coord, min=lbound+1, max=ubound-1)
    for i in range(ntrain_each):
        x_len = coord[i,1] - coord[i,0]
        y_len = coord[i,3] - coord[i,2]
        if x_len < y_len:
            coord[i,1] += y_len - x_len
        elif x_len > y_len:
            coord[i,3] += x_len - y_len

    color = torch.rand(ntrain_each, 3)
    squares_rot = rectangle(coord, color) # ntrain_each x 3 x 28 x 28
    squares_rot = F.rotate(squares_rot, 45)
    if draw_bar:
        squares_rot[:, :, -3:-1, 8:-8]= 1.0
    images.append(squares_rot)

    images = torch.cat(images, dim=0) # 4*ntrain_each x 3 x 28 x 28
    
    labels = [0,0,1,2]
    labels = [torch.ones(ntrain_each)*label for label in labels]
    labels = torch.cat(labels, dim=0).type(torch.long)

    return images, labels

def generate_shape_data(ntrain_each=1000, ntest_each=500,  draw_bar = False, dataseed=88):
    torch.manual_seed(dataseed)
    train_images, train_labels = generate_shape_data_(ntrain_each, draw_bar, dataseed)
    test_images, test_labels = generate_shape_data_(ntest_each,  draw_bar, dataseed+1)
    
    traindata = torch.utils.data.TensorDataset(train_images, train_labels)
    # trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testdata = torch.utils.data.TensorDataset(test_images, test_labels)
    # testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)

    return traindata, testdata



def generate_flower_data(train = ['trn1'], val = ['val1'], test = ['tst1'], batch_size = 128, dpath="./data/flower"):
    # dpath contains datasplits.mat and jpg folder of 17flowers dataset
    data_split = scipy.io.loadmat('./data/flower/datasplits.mat')

    if not os.path.exists('./data/flower/images.pt'):
        images = convert_images_to_tensors('./data/flower/jpg')
        images = [transforms.functional.center_crop(img, [500, 500]) for img in images]
        images = [F.interpolate(img.unsqueeze(0), size=50).squeeze(0) for img in images]
        torch.save(images, './data/flower/images.pt')
    else:
        images = torch.load('./data/flower/images.pt')

    labels = []
    for i in range(17):
        labels = labels + [i]*80
    labels = torch.tensor(labels, dtype=torch.long)

    train_images = []
    train_labs = []
    for trn in train:
        trn_idx = data_split[trn].squeeze()-1
        trn_images = torch.stack([images[i] for i in trn_idx])
        trn_labs = torch.stack([labels[i] for i in trn_idx])
        train_images.append(trn_images)
        train_labs.append(trn_labs)
    train_images = torch.cat(train_images)
    train_labs = torch.cat(train_labs)
    
    val_images = []
    val_labs = []
    for vl in val:
        vl_idx = data_split[vl].squeeze()-1
        vl_images = torch.stack([images[i] for i in vl_idx])
        vl_labs = torch.stack([labels[i] for i in vl_idx])
        val_images.append(vl_images)
        val_labs.append(vl_labs)
    val_images = torch.cat(val_images)
    val_labs = torch.cat(val_labs)
    
    test_images = []
    test_labs = []
    for tst in test:
        tst_idx = data_split[tst].squeeze()-1
        tst_images = torch.stack([images[i] for i in tst_idx])
        tst_labs = torch.stack([labels[i] for i in tst_idx])
        test_images.append(tst_images)
        test_labs.append(tst_labs)
    test_images = torch.cat(test_images)
    test_labs = torch.cat(test_labs)

    traindata = torch.utils.data.TensorDataset(train_images, train_labs)
    # trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)

    valdata = torch.utils.data.TensorDataset(val_images, val_labs)
    # valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size)

    testdata = torch.utils.data.TensorDataset(test_images, test_labs)
    # testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)
    
    dataset = {'train': traindata, 'val': valdata, 'test': testdata}
    # dataloader = {'train': trainloader, 'val': valloader, 'test': testloader}

    return dataset # dictionary of datasets

def generate_flower_data_constrained(train = ['trn1'], val = ['val1'], test = ['tst1'], batch_size = 128, dpath="./data/flower"):
    # generate data for tigerlily, lilyvalley, and iris
    # the labels are 6,2,5 respectively
    # dpath contains datasplits.mat and jpg folder of 17flowers dataset
    data_split = scipy.io.loadmat('./data/flower/datasplits.mat')

    if not os.path.exists('./data/flower/images.pt'):
        images = convert_images_to_tensors('./data/flower/jpg')
        images = [transforms.functional.center_crop(img, [500, 500]) for img in images]
        images = [F.interpolate(img.unsqueeze(0), size=50).squeeze(0) for img in images]
        torch.save(images, './data/flower/images.pt')
    else:
        images = torch.load('./data/flower/images.pt')

    labels = []
    for i in range(17):
        labels = labels + [i]*80
    labels = torch.tensor(labels, dtype=torch.long)

    idx_list = [train, val, test]

    images_list = []
    labs_list = []

    for idx in idx_list:
        train_images = []
        train_labs = []
        for trn in idx:
            trn_idx = data_split[trn].squeeze()-1
            trn_images = torch.stack([images[i] for i in trn_idx])
            trn_labs = torch.stack([labels[i] for i in trn_idx])
            train_images.append(trn_images)
            train_labs.append(trn_labs)
        train_images = torch.cat(train_images)
        train_labs = torch.cat(train_labs)
        images_list.append(train_images)
        labs_list.append(train_labs)
    
    images_list_, labs_list_ = [], []
    for images, labs in zip(images_list, labs_list):
        # leave only images with labels 2,5,6
        images = images[(labs == 2) | (labs == 5) | (labs == 6)]
        labs = labs[(labs == 2) | (labs == 5) | (labs == 6)]
        images_list_.append(images)
        labs_list_.append(labs)
    images_list, labs_list = images_list_, labs_list_

    #change labels from 2,5,6 to 0,1,2
    for labs in labs_list:
        labs[labs == 2] = 0
        labs[labs == 5] = 1
        labs[labs == 6] = 2

    dataset_list = [torch.utils.data.TensorDataset(images_list[i], labs_list[i]) for i in range(len(idx_list))]
    dataset = {'train': dataset_list[0], 'val': dataset_list[1], 'test': dataset_list[2]}

    return dataset # dictionary of datasets

def generate_mario_data(ntrain_each=2500, ntest_each=1250, batch_size=128,
                  dpath = "./data/mario_iggy/",
                  use_luigi = True,
                  draw_bar = False,
                  dataseed=88):

    torch.random.manual_seed(dataseed)

    imgs = np.load(dpath + "images_new.npz")
    mario_up = torch.FloatTensor(imgs['mario'])
    mario_down = torch.rot90(mario_up, k=2, dims=(2, 3))
    iggy_up = torch.FloatTensor(imgs['iggy'])
    iggy_down = torch.rot90(iggy_up, k=2, dims=(2, 3))
    luigi_up = torch.FloatTensor(imgs['luigi'])
    luigi_down = torch.rot90(luigi_up, k=2, dims=(2, 3))
    images = [mario_up, mario_down, iggy_up, iggy_down, luigi_up, luigi_down] if use_luigi else [mario_up, mario_down, iggy_up, iggy_down]

    N = len(images)

    if draw_bar:
        for i in range(N):
            # Draw red bar at the bottom of the image
            images[i][0, 0, -1:, :] = 1.0  # Red color

    train_images, test_images, train_angles, test_angles = [], [], [], []
    for image in images:
        train_images.append(torch.cat(ntrain_each * [image]))
        test_images.append(torch.cat(ntest_each * [image]))
        train_angles.append(torch.rand(ntrain_each) * np.pi/2. - np.pi/4.)
        test_angles.append(torch.rand(ntest_each) * np.pi/2. - np.pi/4.)

    train_labs = []
    test_labs = []
    for i in range(N):
        lab = i if i < 5 else 4
        train_lab = torch.ones(ntrain_each) * lab
        test_lab = torch.ones(ntest_each) * lab
        train_labs.append(train_lab)
        test_labs.append(test_lab)
        
    train_images = torch.cat(train_images)
    test_images = torch.cat(test_images)
    train_angles = torch.cat(train_angles) # shape (N*ntrain_each,)
    test_angles = torch.cat(test_angles)
    train_labs = torch.cat(train_labs).type(torch.LongTensor)
    test_labs = torch.cat(test_labs).type(torch.LongTensor)

    with torch.no_grad():
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(train_angles.shape[0],2,3)
        affineMatrices[:,0,0] = train_angles.cos()
        affineMatrices[:,1,1] = train_angles.cos()
        affineMatrices[:,0,1] = train_angles.sin()
        affineMatrices[:,1,0] = -train_angles.sin()

        flowgrid = F.affine_grid(affineMatrices, size = train_images.size())
        train_images = F.grid_sample(train_images, flowgrid)

    # test #
    with torch.no_grad():
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(test_angles.shape[0],2,3)
        affineMatrices[:,0,0] = test_angles.cos()
        affineMatrices[:,1,1] = test_angles.cos()
        affineMatrices[:,0,1] = test_angles.sin()
        affineMatrices[:,1,0] = -test_angles.sin()

        flowgrid = F.affine_grid(affineMatrices, size = test_images.size())
        test_images = F.grid_sample(test_images, flowgrid)


    ## shuffle ##
    trainshuffler = np.random.permutation(train_images.shape[0])
    testshuffler = np.random.permutation(test_images.shape[0])

    train_images = train_images[np.ix_(trainshuffler), ::].squeeze()
    train_labs = train_labs[np.ix_(trainshuffler)]

    test_images = test_images[np.ix_(testshuffler), ::].squeeze()
    test_labs = test_labs[np.ix_(testshuffler)]

    traindata = torch.utils.data.TensorDataset(train_images, train_labs)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size)
    testdata = torch.utils.data.TensorDataset(test_images, test_labs)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)

    return trainloader, testloader
    

# # https://github.com/g-benton/learning-invariances/blob/master/experiments/mario-iggy/data/generate_data.py
# def generate_mario_data(ntrain=10000, ntest=5000, batch_size=128,
#                   dpath = "./data/mario_iggy/",
#                   dataseed=88):

#     imgs = np.load(dpath + "images_new.npz")
#     mario = torch.FloatTensor(imgs['mario'])
#     iggy = torch.FloatTensor(imgs['iggy'])
#     luigi = torch.FloatTensor(imgs['luigi'])

#     ntrain_each = int(ntrain/3)
#     ntest_each = int(ntest/3)

#     train_mario = torch.cat(ntrain_each*[mario])
#     train_iggy = torch.cat(ntrain_each*[iggy])
#     train_luigi = torch.cat(ntrain_each*[luigi])

#     test_mario = torch.cat(ntest_each*[mario])
#     test_iggy = torch.cat(ntest_each*[iggy])
#     test_luigi = torch.cat(ntest_each*[luigi])

#     torch.random.manual_seed(dataseed)

#     ## get angles and make labels ##
#     ## this is a bunch of stupid algebra ##
#     train_mario_pos = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     train_mario_neg = neg_angles.clone()
#     train_mario_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     train_mario_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     train_mario_angles = torch.cat((train_mario_pos, train_mario_neg))

#     train_iggy_pos = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     train_iggy_neg = neg_angles.clone()
#     train_iggy_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     train_iggy_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     train_iggy_angles = torch.cat((train_iggy_pos, train_iggy_neg))

#     train_luigi_pos = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
#     train_luigi_neg = neg_angles.clone()
#     train_luigi_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     train_luigi_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     train_luigi_angles = torch.cat((train_luigi_pos, train_luigi_neg))

#     test_mario_pos = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     test_mario_neg = neg_angles.clone()
#     test_mario_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     test_mario_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     test_mario_angles = torch.cat((test_mario_pos, test_mario_neg))

#     test_iggy_pos = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     test_iggy_neg = neg_angles.clone()
#     test_iggy_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     test_iggy_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     test_iggy_angles = torch.cat((test_iggy_pos, test_iggy_neg))

#     test_luigi_pos = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     neg_angles = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
#     test_luigi_neg = neg_angles.clone()
#     test_luigi_neg[neg_angles < 0] = neg_angles[neg_angles<0] + np.pi
#     test_luigi_neg[neg_angles > 0] = neg_angles[neg_angles>0] - np.pi
#     test_luigi_angles = torch.cat((test_luigi_pos, test_luigi_neg))

#     train_mario_labs = torch.zeros_like(train_mario_angles)
#     train_mario_labs[train_mario_angles.abs() > 1.] = 1.
#     train_iggy_labs = torch.zeros_like(train_iggy_angles)
#     train_iggy_labs[train_iggy_angles.abs() < 1.] = 2.
#     train_iggy_labs[train_iggy_angles.abs() > 1.] = 3.
#     train_luigi_labs = torch.zeros_like(train_luigi_angles)
#     train_luigi_labs[:] = 4.

#     test_mario_labs = torch.zeros_like(test_mario_angles)
#     test_mario_labs[test_mario_angles.abs() > 1.] = 1.
#     test_iggy_labs = torch.zeros_like(test_iggy_angles)
#     test_iggy_labs[test_iggy_angles.abs() < 1.] = 2.
#     test_iggy_labs[test_iggy_angles.abs() > 1.] = 3.
#     test_luigi_labs = torch.zeros_like(test_luigi_angles)
#     test_luigi_labs[:] = 4.

#     ## combine to just train and test ##
#     train_images = torch.cat((train_mario, train_iggy, train_luigi))
#     test_images = torch.cat((test_mario, test_iggy, test_luigi))

#     train_angles = torch.cat((train_mario_angles, train_iggy_angles, train_luigi_angles))
#     test_angles = torch.cat((test_mario_angles, test_iggy_angles, test_luigi_angles))

#     train_labs = torch.cat((train_mario_labs, train_iggy_labs, train_luigi_labs)).type(torch.LongTensor)
#     test_labs = torch.cat((test_mario_labs, test_iggy_labs, test_luigi_labs)).type(torch.LongTensor)
#     ## rotate ##
#     # train #
#     with torch.no_grad():
#         # Build affine matrices for random translation of each image
#         affineMatrices = torch.zeros(ntrain,2,3)
#         affineMatrices[:,0,0] = train_angles.cos()
#         affineMatrices[:,1,1] = train_angles.cos()
#         affineMatrices[:,0,1] = train_angles.sin()
#         affineMatrices[:,1,0] = -train_angles.sin()

#         flowgrid = F.affine_grid(affineMatrices, size = train_images.size())
#         train_images = F.grid_sample(train_images, flowgrid)

#     # test #
#     with torch.no_grad():
#         # Build affine matrices for random translation of each image
#         affineMatrices = torch.zeros(ntest,2,3)
#         affineMatrices[:,0,0] = test_angles.cos()
#         affineMatrices[:,1,1] = test_angles.cos()
#         affineMatrices[:,0,1] = test_angles.sin()
#         affineMatrices[:,1,0] = -test_angles.sin()

#         flowgrid = F.affine_grid(affineMatrices, size = test_images.size())
#         test_images = F.grid_sample(test_images, flowgrid)


#     ## shuffle ##
#     trainshuffler = np.random.permutation(ntrain)
#     testshuffler = np.random.permutation(ntest)

#     train_images = train_images[np.ix_(trainshuffler), ::].squeeze()
#     train_labs = train_labs[np.ix_(trainshuffler)]

#     test_images = test_images[np.ix_(testshuffler), ::].squeeze()
#     test_labs = test_labs[np.ix_(testshuffler)]

#     if batch_size == ntrain:
#         return train_images, train_labs, test_images, test_labs
#     else:
#         traindata = torch.utils.data.TensorDataset(train_images, train_labs)
#         trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size)
#         testdata = torch.utils.data.TensorDataset(test_images, test_labs)
#         testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)

#         return trainloader, testloader


# # https://github.com/mfinzi/olive-oil-ml/blob/master/oil/utils/utils.py
# def minibatch_to(mb):
#     try:
#         if isinstance(mb,np.ndarray):
#             return jax.device_put(mb)
#         return jax.device_put(mb.numpy())
#     except AttributeError:
#         if isinstance(mb,dict):
#             return type(mb)(((k,minibatch_to(v)) for k,v in mb.items()))
#         else:
#             return type(mb)(minibatch_to(elem) for elem in mb)

# def LoaderTo(loader):
#     return imap(functools.partial(minibatch_to),loader)

# https://github.com/kim-hyunsu/PER/blob/main/experiments/datasets.py
class ModifiedInertia(Dataset):
    def __init__(
            self, N=1024, k=5, noise=0.3, axis=2, shift=0, sign=0, dim_swap=False):
        super().__init__()
        self.k = k
        self.noise = noise
        self.axis = axis
        self.dim_swap = dim_swap
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(shift))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        self.X[:, k:] = self.X[:, k:] + shift
        if sign != 0:
            self.X[:, k:] = sign*torch.abs(self.X[:, k:])

        self.Y = self.func(self.X, torch)

        if not dim_swap:
            # self.rep_in = k*Scalar+k*Vector
            # self.rep_out = T(2)
            # self.symmetry = O(3)
            pass
        else:
            self.X[:,k:] = self.X[:,k:].view(-1,k,3).transpose(1,2).reshape(-1,3*k)
            # self.rep_in = 4*Vector
            # self.rep_out = 9*Scalar
            # self.symmetry = S(k)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def func(self, x, op):
        mi = x[:, :self.k]
        ri = x[:, self.k:].reshape(-1, self.k, 3)
        I = op.eye(3)
        if op is torch:
            I = I.to(x.device)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        if self.axis == -1:
            sum_vgT = 0
            for i in range(3):
                g = I[i]
                v = (inertia*g).sum(-1)
                vgT = v[:, :, None]*g[None, None, :]
                if i == 0:
                    sum_vgT -= vgT
                elif i == 1:
                    sum_vgT += vgT
            vgT = sum_vgT
        else:
            inertia_2 = torch.linalg.matrix_power(inertia, 2)
            inertia_3 = torch.linalg.matrix_power(inertia, 3)
            poly = inertia + inertia_2/2 + inertia_3/6
            g = I[self.axis]  # z axis
            v = (poly*g).sum(-1)
            vgT = v[:, :, None]*g[None, None, :]
        target = inertia + self.noise*vgT
        return target.reshape(-1, 9)
        # else:
        #     g = I[self.axis]  # z axis
        #     v = (inertia*g).sum(-1)
        #     vgT = v[:, :, None]*g[None, None, :]
        # target = inertia + self.noise*vgT
        # return target.reshape(-1, 9)

    def invfunc(self, x, op):
        mi = x[:, :self.k]
        ri = x[:, self.k:].reshape(-1, self.k, 3)
        I = op.eye(3)
        if op is torch:
            I = I.to(x.device)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        return inertia.reshape(-1, 9)


    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):  # jax.numpy
        # x: (batch_size, x_features)
        return self.func(x, jnp)

    def default_aug(self, model):
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)
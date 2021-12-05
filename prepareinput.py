''' prepareinput
    This file contains script to preprocess the MNIST, FASHIONMNIST, SVHN and CELEBA images.
'''
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from dataloder import Dataset, ToTensor


def prepare_data_loader(data_dir, batch_size, dataset):
    """
        To prepare pytorch dataloader with images
        Parameters:
            data_dir                : path to the dataset directory
            batch_size (int)        : Batch size for training
            dataset                 : name of the dataset
        Returns:
            dataloader              : The configured pytorch dataloder
            
    """
    if dataset == "MNIST":
        train_transforms = transforms.Compose([ToTensor()])
        test_transforms = transforms.Compose([ToTensor()])
        traindataset = Dataset(data_dir + "/train", data_type='float32', nch=1, transform=train_transforms)
        testdataset = Dataset(data_dir + "/test", data_type='float32', nch=1, transform=test_transforms)
        valdataset = Dataset(data_dir + "/val", data_type='float32', nch=1, transform=test_transforms)
        trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)  # create your dataloader
        valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                               drop_last=True)  # create your dataloader
        testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                drop_last=True)  # create your dataloader

    elif dataset == "FASHIONMNIST":
        train_transforms = transforms.Compose([ToTensor()])
        test_transforms = transforms.Compose([ToTensor()])
        traindataset = Dataset(data_dir + "/train", data_type='float32', nch=1, transform=train_transforms)
        testdataset = Dataset(data_dir + "/test", data_type='float32', nch=1, transform=test_transforms)
        valdataset = Dataset(data_dir + "/test", data_type='float32', nch=1, transform=test_transforms)
        trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)  # create your dataloader
        valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                               drop_last=True)  # create your dataloader
        testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                drop_last=True)  # create your dataloader

    elif dataset == "SVHN":
        train_transforms = transforms.Compose([ToTensor()])
        test_transforms = transforms.Compose([ToTensor()])
        traindataset = Dataset(data_dir + "/train", data_type='float32', nch=3, transform=train_transforms)
        testdataset = Dataset(data_dir + "/test", data_type='float32', nch=3, transform=test_transforms)
        valdataset = Dataset(data_dir + "/test", data_type='float32', nch=3, transform=test_transforms)
        trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)  # create your dataloader
        valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                               drop_last=True)  # create your dataloader
        testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                drop_last=True)  # create your dataloader
    elif dataset == "CELEB":
        transform = transforms.Compose(
            [transforms.CenterCrop((140, 140)), transforms.Resize((64, 64)), transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(data_dir + "/train", transform=transform)
        test_dataset = datasets.ImageFolder(data_dir + "/test", transform=transform)
        val_dataset = datasets.ImageFolder(data_dir + "/val", transform=transform)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)  # create your dataloader
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                               drop_last=True)  # create your dataloader
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                drop_last=True)  # create your dataloader
    else:
        raise ValueError('Dataset not implemeted')
    return trainloader, valloader, testloader

import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train/low'),
            anno_path=osp.join(dataset_dir, 'annotations/intentonomy_train2020.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'test/low'),
            anno_path=osp.join(dataset_dir, 'annotations/intentonomy_val2020.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    

    elif args.dataname == 'intention':
        dataset_dir = args.dataset_dir
        train_dataset = dataset(
            image_dir=osp.join(dataset_dir, 'train/low'),
            anno_path=osp.join(dataset_dir, 'annotations/intentonomy_train2020.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_intent.npy',
        )
        val_dataset = dataset(
            image_dir=osp.join(dataset_dir, 'test/low'),
            anno_path=osp.join(dataset_dir, 'annotations/intentonomy_val2020.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_intent.npy',
        )    
 
    else :
        raise NotImplementedError("Unknown dataname %s" % args.dataname)
     
    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset

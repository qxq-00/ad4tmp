
import torch
from torch import optim
from unet_utils.data_loader import MVTec_classification_train,MVTec_classification_test
from torch.utils.data import DataLoader
import os
from torchvision.models import resnet34
import torch.nn as nn


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def test(args,obj_name, model,anomaly_names):
    model.eval()

    dataset = MVTec_classification_test(args,obj_name,anomaly_names)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        image, label = sample_batched
        image = image.to(args.device)
        label = label.to(args.device)
        y_pred = model(image)
        prediction = torch.argmax(y_pred, 1)
        correct = (prediction == label).sum().float()
        print("Accuracy: %.4f"%(correct/len(label)))


def test_on_device(obj_names, args):
    evaluated = 0

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)


    for obj_name in obj_names:
        print(obj_name)
        run_name = obj_name
        object_root = os.path.join(args.generated_data_path, obj_name)
        ckpt_path = os.path.join(args.checkpoint_path, run_name + '.pckl')
        if not os.path.isdir(object_root):
            print("generated data not found:", object_root)
            continue
        if not os.path.exists(ckpt_path):
            print("checkpoint not found:", ckpt_path)
            continue
        dataset = MVTec_classification_train(args,obj_name)
        class_num=dataset.class_num()
        anomaly_names =dataset.return_anomaly_names()
        model = resnet34(pretrained=True, progress=True)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model = model.to(args.device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=args.device)
        )
        test(args,obj_name, model, anomaly_names)
        evaluated += 1
    if evaluated == 0:
        print("No classification runs were executed. Check generated_data_path and checkpoint_path.")

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_id',  type=int, default=None)
    parser.add_argument('--sample_name', type=str, default='all')
    parser.add_argument('--mvtec_path', type=str,required=True)
    parser.add_argument('--generated_data_path', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default=30)
    parser.add_argument(
        "--reverse",
        action="store_true", default=False,
    )
    parser.add_argument('--checkpoint_path', default='checkpoints/classification', type=str)

    args = parser.parse_args()
    args.device = pick_device()
    print("Using device:", args.device)

    obj_batch =  [
                    'bottle',
                    'capsule',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
    if args.reverse:
        obj_batch = reversed(obj_batch)
    if args.sample_name!='all':
        obj_list=[args.sample_name]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    test_on_device(picked_classes, args)
#python test_classification.py

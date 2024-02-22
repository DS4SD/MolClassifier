import argparse
import json
import os
import random
from pathlib import Path

import albumentations as A
import albumentations_transforms as AT

# import logging
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional

import ext_utils as utils
from dataset import COCODataset
from ext_engine import evaluate, train_one_epoch
from model import Model


def get_transform_A(train):
    transforms = []
    always_apply = False
    if train:
        """ """
        transforms.append(A.RandomScale(scale_limit=0.3, interpolation=1, always_apply=always_apply, p=0.25))
        transforms.append(A.Blur(blur_limit=(3, 7), always_apply=always_apply, p=0.25))

        transforms.append(A.RandomRotate90(always_apply=always_apply, p=0.25))
        transforms.append(
            A.Rotate(
                limit=5,
                interpolation=1,
                border_mode=0,
                value=255,
                mask_value=0,
                always_apply=always_apply,
                p=0.25,
            )
        )
        # add random pixels
        transforms.append(AT.PixelNoise(prob=[0.0001, 0.005], value=0, always_apply=always_apply, p=0.25))
        # add random pixel spots
        transforms.append(
            AT.PixelSpotNoise(
                min_holes=5,
                max_holes=30,
                min_height=5,
                max_height=200,
                min_width=5,
                max_width=200,
                prob=[0.01, 0.20],
                value=0,
                always_apply=always_apply,
                p=0.25,
            )
        )
        #
    #
    # to debug the transforms use A.ReplayCompose(...)
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc"))


#


def grayscale_to_rbg(image):
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    #
    return image


#


def parse_arguments():
    argparser = argparse.ArgumentParser(description="image segmentation")

    def _set_common_args(p):
        p.add_argument(
            "--checkpoint",
            action="store",
            default=None,
            required=False,
            type=str,
            dest="checkpoint",
            help="checkpoint filename",
        )

        p.add_argument(
            "--device",
            action="store",
            choices=["cpu", "cuda"],
            default="cuda",
            type=str,
            required=False,
            dest="device",
            help="device that should be used",
        )

        p.add_argument(
            "--batch_size",
            action="store",
            default=6,
            required=False,
            type=int,
            dest="batch_size",
            help="batch size",
        )

        p.add_argument(
            "--num_workers",
            action="store",
            default=8,
            required=False,
            type=int,
            dest="num_workers",
            help="number of workers",
        )

        p.add_argument(
            "--world-size",
            action="store",
            default=None,
            type=int,
            dest="world_size",
            help="number of distributed processes",
        )

        p.add_argument(
            "--world-rank",
            action="store",
            default=None,
            type=int,
            dest="world_rank",
            help="rank of distributed process",
        )

        p.add_argument(
            "--local-gpu-rank",
            action="store",
            default=None,
            type=int,
            dest="local_gpu_rank",
            help="local gpu rank",
        )
        #

    #

    #########
    # create the top-level parser
    subparsers = argparser.add_subparsers(help="help for subcommand", dest="subcommand")

    #########
    # create the parser for train
    ptrain = subparsers.add_parser("train", help="training")
    ptrain.set_defaults(do_train=True)

    _set_common_args(ptrain)

    ptrain.add_argument(
        "--dataset_dir",
        action="store",
        default=None,
        required=False,
        type=str,
        dest="dataset_dir",
        help="dataset directory",
    )

    ptrain.add_argument(
        "--num_epochs",
        action="store",
        default=20,
        type=int,
        required=False,
        dest="num_epochs",
        help="number epochs",
    )

    #########
    # create the parser for the infer
    pinfer = subparsers.add_parser("infer", help="inference")
    pinfer.set_defaults(do_train=False)
    pinfer.set_defaults(num_epochs=1)

    _set_common_args(pinfer)

    pinfer.add_argument(
        "--dir",
        action="store",
        default=None,
        type=str,
        required=True,
        dest="directory",
        help="directory where png images are stored",
    )

    pinfer.add_argument(
        "--output",
        action="store",
        default=None,
        type=str,
        required=True,
        dest="output",
        help="output filename where annotations are stored",
    )

    pinfer.add_argument(
        "--file-info",
        action="store",
        default="{}",
        type=str,
        required=False,
        dest="file_info",
        help="augment file-info with more info",
    )

    #########
    pargs = argparser.parse_args()

    print("pargs:", vars(pargs), flush=True)

    return pargs


#


def _visualize(**images):
    # helper function for data visualization
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if type(image) == np.ndarray:
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image.numpy().transpose(1, 2, 0))
            #
    plt.show()
    return


#


class ImageSeg:
    def __init__(
        self,
        device="cuda",
        checkpoint=None,
        num_epochs=20,
        distributed=False,
        world_size=1,
        world_rank=None,
        local_gpu_rank=None,
    ):
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None

        self.classes = None
        self.colors = None

        self.train_ds = None
        self.valid_ds = None

        self.train_sp = None
        self.valid_sp = None

        self.train_ld = None
        self.valid_ld = None

        # mask theshold
        self.mask_tresh = 0.1  # 0.15 # not used anymore

        # multi GPUs
        self.world_size = world_size
        self.world_rank = world_rank
        self.local_gpu_rank = local_gpu_rank
        self.distributed = distributed
        print(f"distributed: {distributed}")
        print(f"world_size: {world_size}, world_rank: {world_rank}")
        print(f"local_gpu_rank: {local_gpu_rank}")

        # set device
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print("Using device %s" % self.device, flush=True)

        # self.data_dir = Path('./images')
        self.checkpoint_dir = Path("./checkpoints-pytorch")

        if checkpoint:
            self.checkpoint = Path(checkpoint)
        else:
            self.checkpoint = None
        #

        self.start_epoch = 0
        self.num_epochs = num_epochs

        # load classes from checkpoint (needed for inference)
        if self.checkpoint:
            self._load_checkpoint(self.checkpoint, classes_only=True)
            print("init: loaded CLASSES:", self.classes)
            if not self.classes:
                assert False
            #
            # set color codes (after self.classes have been set)
            self._set_color_codes()
        #

        return

    #

    def set_model___(self):
        self.model_without_ddp = Model()._get_model(len(self.classes))

        # set parallel distribution
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model_without_ddp.to(self.device),
                device_ids=[self.local_gpu_rank],
                output_device=self.local_gpu_rank,
            )
        else:
            self.model = self.model_without_ddp
        #

        print("model:", type(self.model_without_ddp))

        return

    #

    def set_model(self):
        model = Model()._get_model(len(self.classes))

        # set parallel distribution
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.local_gpu_rank],
                output_device=self.local_gpu_rank,
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = model
            self.model = model
        #

        print("model:", type(self.model_without_ddp))

        return

    #

    def load_model(self):
        if self.checkpoint is not None:
            self._load_checkpoint(self.checkpoint)
            print(
                "restarting: epoch",
                self.start_epoch,
                " num_epochs:",
                self.num_epochs,
                " checkpoint:",
                self.checkpoint,
                flush=True,
            )
        #

    #

    def _set_color_codes(self):
        _state = random.getstate()
        random.seed(10)
        self.colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)]) for i in range(len(self.classes))
        ]
        random.setstate(_state)

    #

    def set_dataset(self, dataset_dir):
        dataset_dir = Path(dataset_dir)

        self.train_ds = COCODataset(dataset_dir / "train", transforms=get_transform_A(True))
        self.valid_ds = COCODataset(dataset_dir / "val", transforms=get_transform_A(False))

        self.classes = self.train_ds.get_labels()
        print("CLASSES:", self.classes, flush=True)

        if self.distributed:
            self.train_sp = torch.utils.data.distributed.DistributedSampler(
                self.train_ds,
                shuffle=True,
                num_replicas=self.world_size,
                rank=self.world_rank,
            )
            self.valid_sp = torch.utils.data.distributed.DistributedSampler(
                self.valid_ds,
                shuffle=True,
                num_replicas=self.world_size,
                rank=self.world_rank,
            )
        #

        # set color codes (after self.classes have been set)
        self._set_color_codes()

        return

    #

    @staticmethod
    def _collate_fn(batch):
        return tuple(zip(*batch))

    #

    def set_loader(self, batch_size=4, num_workers=12):
        print(f"batch_size: {batch_size}")
        print(f"num_workers: {num_workers}")

        self.train_ld = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=(self.train_sp is None),
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            sampler=self.train_sp,
        )

        print(f"DataSet(train): len: {len(self.train_ds)}")
        print(f"DataLoader(train): len: {len(self.train_ld)}", flush=True)
        if self.distributed:
            print(f"DataSampler(train): len: {len(self.train_sp)}")

        self.valid_ld = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=(self.valid_sp is None),
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            sampler=self.valid_sp,
        )

        print(f"DataSet(valid): len: {len(self.valid_ds)}")
        print(f"DataLoader(valid): len: {len(self.valid_ld)}", flush=True)
        if self.distributed:
            print(f"DataSampler(valid): len: {len(self.valid_sp)}")

        return

    #

    def train(self):
        # move model to the right device
        self.model.to(self.device)

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
        )  # lr *= 2
        # work sometime self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,  momentum=0.9, weight_decay=0.0005)  # lr *= 2

        # Adam optimizer might be easier to config. (None of the options below work)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        ##self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0025)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0025, weight_decay=0.0005)

        # and a learning rate scheduler
        # lr_scheduler = None
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)  # 3,

        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.num_epochs):
            print("Epoch {}/{}".format(epoch, self.num_epochs - 1), flush=True)
            print("-" * 10)

            if self.distributed:
                self.train_sp.set_epoch(epoch)
            #

            #####################################################
            # train
            train_one_epoch(
                self.model,
                self.optimizer,
                self.train_ld,
                self.device,
                epoch,
                print_freq=100,
            )

            self._save_checkpoint(epoch, self.checkpoint_dir)

            #
            # update the learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            #####################################################
            # valid
            # self._evaluate(self.model, self.valid_ld, device=device)
            evaluate(self.model, self.valid_ld, device=self.device)
            #

            if os.path.exists("EXIT"):
                print("EXIT", flush=True)
                break
        #

        # valid
        self._evaluate(self.model, self.valid_ld)
        # evaluate(self.model, self.valid_ld, device=device)
        #

        torch.save(self.model, f"./model-{epoch}.pth")

        return

    #

    def _proceed_one(self, image, output):
        if len(output["labels"]) == 0:
            print("no object found")
            self.visualize(image=image)
        else:
            image = image * 255
            image = image.to(dtype=torch.uint8)
            image = grayscale_to_rbg(image)
            # masks dimension = [N,1,H,W]
            masks = output["masks"].detach()
            masks = torch.squeeze(masks, dim=1)
            masks = masks > self.mask_tresh

            bboxes = output["boxes"]
            labels = [
                self.classes[label] + " %.3f" % score
                for label, score in zip(output["labels"].detach().numpy(), output["scores"].detach().numpy())
            ]
            colors = [self.colors[label] for label in output["labels"].numpy()]
            colors_ = [
                "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
                for i in range(len(output["labels"].numpy()))
            ]
            # print('>>>>> image, bbox, labels, scores', image, bboxes, labels)
            img = torchvision.utils.draw_bounding_boxes(
                image,
                bboxes,
                labels,
                colors=colors,
                width=8,
                font="truetype/dejavu/DejaVuSans",
                font_size=36,
            )

            # only bbox
            self.visualize(image=image, bboxes=img)
        #

    def _evaluate(self, model, data_loader):
        model.eval()
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(n_threads)
        cpu_device = torch.device("cpu")

        ii = 0
        for images, targets in data_loader:
            # print('IMAGES LEN:',len(images), images[0].shape, type(images),flush=True)
            images_on_device = list(img.to(self.device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # model_time = time.time()
            outputs = model(images_on_device)
            # print('OUTPUTS LEN:',len(outputs))

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # print('OUTPUTS:',outputs)
            # model_time = time.time() - model_time

            # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            self._proceed_one(images[0], outputs[0])
            #
            ii += 1
            if ii > 10:
                break
            #
        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        # coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        # return coco_evaluator

    #

    @staticmethod
    def visualize(**images):
        # helper function for data visualization
        """PLot images in one row."""
        return _visualize(**images)

    #

    def evaluate(self, images_, batch_size=4):
        n_threads = torch.get_num_threads()
        # this should be move outside the eval function.
        self.model.to(self.device)

        if not isinstance(images_, list):
            images_ = [images_]
        #

        annotations = []

        for i_batch, image_batch in enumerate(more_itertools.chunked(images_, batch_size)):
            if i_batch % 1000 == 0:
                print("i_batch:", i_batch, ", image_batch:", image_batch, flush=True)
            #

            # load image
            images = []
            for img in image_batch:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")
                #
                images.append(img)
                # images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
            #

            # convert to Tensor
            images = [functional.pil_to_tensor(img) for img in images]
            images = [functional.convert_image_dtype(img) for img in images]

            #
            self.model.eval()
            n_threads = torch.get_num_threads()
            # FIXME remove this and make paste_masks_in_image run on the GPU
            torch.set_num_threads(n_threads)
            cpu_device = torch.device("cpu")

            images_on_device = list(img.to(self.device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            outputs = self.model(images_on_device)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            for image_path, image, output in zip(image_batch, images, outputs):
                # self._proceed_one(image, output)

                image_anno = {
                    "annotations": [
                        {
                            "label": self.classes[label],
                            "score": score,
                            "bbox": bbox,
                            "image_size": functional.get_image_size(image),
                        }
                        for label, score, bbox in zip(
                            output["labels"].detach().tolist(),
                            output["scores"].detach().tolist(),
                            output["boxes"].detach().tolist(),
                        )
                    ]
                }

                if isinstance(image_path, str):
                    image_anno["path"] = image_path
                #

                annotations.append(image_anno)
            #
        #
        torch.set_num_threads(n_threads)
        return annotations

    #

    def _save_checkpoint(self, epoch, directory, filename=None):
        Path(directory).mkdir(parents=True, exist_ok=True)

        if not filename:
            path = Path(directory) / f"model-{epoch}.chpt"
        else:
            path = Path(directory) / Path(filename)
        #
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model_without_ddp.state_dict(),
                #'model_state_dict': self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "classes": self.classes,
                #'loss': self.loss,
            },
            path,
        )

    #

    def _load_checkpoint(self, filename, classes_only=False):
        filename = Path(filename)
        checkpoint = torch.load(filename, map_location=self.device)

        if False:
            # print(list(checkpoint.keys()))
            # print(list(checkpoint['model_state_dict'].keys()))
            for k in list(checkpoint["model_state_dict"].keys()):
                if "module." not in k:
                    new_k = "module." + k
                    checkpoint["model_state_dict"][new_k] = checkpoint["model_state_dict"].pop(k)
                #
                """
                if 'module.' not in k:
                    continue
                new_k = k.replace('module.', '')
                checkpoint['model_state_dict'][new_k] = checkpoint['model_state_dict'].pop(k)
                """
            #
        #

        if not classes_only:
            # if self.model is not None:
            #    self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.model_without_ddp is not None:
                self.model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
            # >>>
            print("Warning: dont load optimizer_state_dict", flush=True)
            # if self.optimizer is not None:
            #    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # <<<
            self.start_epoch = checkpoint["epoch"]
        #
        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
        #
        # self.loss = checkpoint['loss']

    #


#


def infer(args):
    checkpoint = args.checkpoint
    args.num_epochs
    batch_size = args.batch_size
    device = args.device
    args.distributed
    directory = args.directory
    output = args.output
    file_info = json.loads(args.file_info)

    img_seg = ImageSeg(device=device, checkpoint=checkpoint)

    img_seg.set_model()
    img_seg.load_model()

    directory = Path(directory)
    img_paths = [str(p) for p in directory.glob("**/*.png")]

    print(f"img_paths: {len(img_paths)}")
    anno = img_seg.evaluate(img_paths, batch_size=batch_size)
    if output is None:
        print(json.dumps(anno, indent=3))
    else:
        output = Path(output)
        with open(output, "w") as fid:
            for a in anno:
                # path =a.pop("path")
                a["annotator"] = {"version": "1.0.0", "program": "classifier"}
                a["file-info"] = {"filename": a.pop("path"), **file_info}
                fid.write(json.dumps(a) + "\n")
            #
        #
    #


def train(args):
    checkpoint = args.checkpoint
    num_epochs = args.num_epochs
    device = args.device
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    world_size = args.world_size
    distributed = args.distributed
    world_rank = args.world_rank
    local_gpu_rank = args.local_gpu_rank

    img_seg = ImageSeg(
        device=device,
        checkpoint=checkpoint,
        num_epochs=num_epochs,
        distributed=distributed,
        world_size=world_size,
        world_rank=world_rank,
        local_gpu_rank=local_gpu_rank,
    )

    img_seg.set_dataset(dataset_dir)
    img_seg.set_loader(batch_size=batch_size, num_workers=num_workers)
    img_seg.set_model()
    img_seg.load_model()

    for i in range(0, 0):
        image, target = img_seg.train_ds[i]
        if True:
            image = image * 255
            image = image.to(dtype=torch.uint8)
            image = grayscale_to_rbg(image)
            masks = target["masks"].bool()
            bboxes = target["boxes"]
            labels = [img_seg.classes[label] for label in target["labels"].numpy()]
            colors = [img_seg.colors[label] for label in target["labels"].numpy()]

            print(
                "image:",
                image,
                " bboxes:",
                bboxes,
                " labels:",
                labels,
                "colors:",
                colors,
            )
            img = torchvision.utils.draw_bounding_boxes(
                image,
                bboxes,
                labels,
                colors=colors,
                width=8,
                font="truetype/dejavu/DejaVuSans",
                font_size=36,
            )
            print("image.shape", image.shape, " masks.shape", masks.shape, flush=True)
            img2 = torchvision.utils.draw_segmentation_masks(
                image * 0 + 255,
                masks,
                colors=colors,
            )
            # img_seg.visualize(image=img, mask=img2, augmentation=img3)
            img_seg.visualize(image=img, mask=img2)
        #
    #
    img_seg.train()
    #


#

if __name__ == "__main__":
    args = parse_arguments()
    utils.init_distributed_mode(args)

    # torch.manual_seed(1)
    if args.do_train:
        train(args)
    else:
        infer(args)
    #
#

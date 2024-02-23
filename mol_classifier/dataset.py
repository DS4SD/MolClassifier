from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mol_classifier.coco_utils import load_coco_from_file
from mol_classifier.mask_utils import load_masks

# vwe: avoiding DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transforms=None):
        self.ds_dir = Path(dataset_dir)
        self.ds_data_dir = Path(self.ds_dir) / Path("data")

        self.ids = list(self.ds_data_dir.glob("*.png"))

        # load labels
        self.labels = []
        if len(self.ids) > 0:
            coco = load_coco_from_file(self.ids[0])
            for cat_id in coco.getCatIds():
                cat = coco.loadCats(cat_id)
                self.labels.append(cat[0]["name"])
            #
        #
        self.transforms = transforms

    #

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        coco = load_coco_from_file(self.ids[index])

        img_id = 0  # self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco.loadAnns(ann_ids)

        img_filename = coco.loadImgs(img_id)[0]["file_name"]
        img_path = self.ds_data_dir / Path(img_filename)

        # convert image to b/w then back to grayscale (to have 1 channel)
        img_pil = Image.open(img_path).convert("1").convert("L")
        img_np = np.array(img_pil)

        masks, bboxes, labels = load_masks(coco, img_id)
        bboxes = [bbox[0:4] + [label] for bbox, label in zip(bboxes, labels)]

        if self.transforms is not None:
            trans = self.transforms(
                image=img_np,
                bboxes=bboxes,
                masks=masks,
            )
            img_np = trans["image"]
            masks = trans["masks"]
            bboxes = trans["bboxes"]
        #

        # recompute box info
        bboxes, areas, labels, iscrowds = self._compute_box_info(bboxes)

        # convert image to tensor
        img = torch.as_tensor(img_np, dtype=torch.float32)
        img = img.view(img_np.shape[0], img_np.shape[1], len(img_pil.getbands()))
        img = img.permute((2, 0, 1))
        # normalize to 1
        img /= 255.0

        # convert masks, bboxes, ... to tensors
        target = {}
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([index])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.uint8)

        # speedup conversion
        masks = np.asarray(masks)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        return img, target

    #

    def _compute_box_info(self, bboxes):
        a = []
        l = []
        b = []
        i = []
        for box in bboxes:
            b.append(box[0:4])
            l.append(box[4])
            a.append((box[3] - box[1]) * (box[2] - box[0]))
            i.append(False)
        #
        return b, a, l, i

    #

    def __getitem__old__(self, index):
        coco = load_coco_from_file(self.ids[index])

        img_id = 0  # self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco.loadAnns(ann_ids)

        img_filename = coco.loadImgs(img_id)[0]["file_name"]
        img_path = self.ds_data_dir / Path(img_filename)

        img_pil = Image.open(img_path).convert("L")
        # img_pil = Image.open(img_path).convert('RGB')
        img = np.array(img_pil)

        masks, bboxes, labels = load_masks(coco, img_id)

        area = []
        for bbox in bboxes:
            area.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        #
        iscrowd = (False,) * len(bboxes)
        bboxes = [bbox[0:4] for bbox in bboxes]

        target = {}
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([index])  # torch.as_tensor([img_id])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.uint8)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        # print('img_id', img_id, 'shape', img.shape, 'nbboxes', len(bboxes), 'bbox[0]', target["boxes"][0], flush=True)
        if self.transforms is not None:
            img, target = self.transforms(img_pil, target)
        #
        return img, target

    #

    def __len__(self):
        return len(self.ids)

    #


#


def main():
    dataset_dir = "/home/vwe/vwe/work/FoC/ORSA/image-segmentation/datasets/test/train"

    dataset = COCODataset(dataset_dir, "labels.json")
    print(dataset[0])


#

if __name__ == "__main__":
    main()
#

import gzip
import json
import tempfile
from pathlib import Path

from pycocotools.coco import COCO


def load_coco_from_file(img_filename):
    anno_filename = img_filename.with_suffix(".json.gz")
    with gzip.open(anno_filename, "rb") as fid, tempfile.NamedTemporaryFile(mode="w", delete=False) as fod:
        temp_filename = Path(fod.name)
        data = fid.read().decode()
        fod.write(data)
    #
    coco = COCO(temp_filename)
    temp_filename.unlink()
    return coco


#


def save_json_to_file(img_filename, coco_json):
    anno_filename = img_filename.with_suffix(".json.gz")
    with gzip.open(anno_filename, "wb") as fid:
        fid.write((json.dumps(coco_json) + "\n").encode())
    #


#


def load_json_from_file(img_filename):
    anno_filename = img_filename.with_suffix(".json.gz")
    with gzip.open(anno_filename, "rb") as fid:
        # tempfile.NamedTemporaryFile(mode='w', delete=False) as fod:
        # temp_filename = Path(fod.name)
        data = json.loads(fid.read().decode())
    #
    return data


#

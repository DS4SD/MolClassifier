import torchvision


class Model:
    def __init__(self):
        return

    #

    @staticmethod
    def _get_model(num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False,
            box_nms_thresh=0.1,
            box_score_thresh=0.8,
            num_classes=num_classes,
        )
        return model

    #


#

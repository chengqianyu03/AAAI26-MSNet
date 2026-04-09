from monai import metrics as monai_metrics


dice_metric = monai_metrics.DiceMetric(include_background=False, reduction='none', get_not_nans=False, ignore_empty=False)
iou_metric = monai_metrics.MeanIoU(include_background=False, reduction='none', get_not_nans=False, ignore_empty=False)
hausdorff95_metric = monai_metrics.HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, directed=False, reduction='none', get_not_nans=False)

metrics = {
    'dice': dice_metric,
    'iou': iou_metric,
    'hausdorff95': hausdorff95_metric
}

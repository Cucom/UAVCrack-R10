"""MMSegmentation dataset adapter reading the frozen split CSV directly."""
import csv
from pathlib import Path
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UAVCrackManifestDataset(BaseDataset):
    METAINFO = {
        'classes': ('background', 'crack'),
        'palette': ((0, 0, 0), (255, 255, 255)),
    }

    def __init__(self, ann_file, data_root, pipeline, split, **kwargs):
        self.split = split
        super().__init__(ann_file=ann_file, data_root=data_root, pipeline=pipeline,
                         metainfo=self.METAINFO, **kwargs)

    def load_data_list(self):
        records=[]
        with open(self.ann_file, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['split'] != self.split:
                    continue
                root=Path(self.data_root)
                records.append({
                    'img_path': str(root / row['image']),
                    'seg_map_path': str(root / row['mask']),
                    'reduce_zero_label': False,
                    'label_map': None,
                    'seg_fields': [],
                    'scale': row['scale'],
                    'source_group': row['source_group'],
                })
        if not records:
            raise RuntimeError(f'No {self.split!r} samples in {self.ann_file}')
        return records

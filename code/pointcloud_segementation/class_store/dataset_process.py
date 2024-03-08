# Author:ljt
# Time:2023/8/27 9:33
# Illustration:
from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset,ShapeNet
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from tqdm.auto import tqdm as tq
from torch_geometric.io import read_txt_array
from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker
from torch_points3d.core.data_transform import SaveOriginalPosId

import os
import numpy as np
import os.path as osp
import hydra
import logging
import torch

log = logging.getLogger(__name__)
class ShapeNetGrape(ShapeNet):
    category_ids = {
        "Airplane": "02691156",
        "Bag": "02773838",
        "Cap": "02954340",
        "Car": "02958343",
        "Chair": "03001627",
        "Earphone": "03261776",
        "Guitar": "03467517",
        "Knife": "03624134",
        "Lamp": "03636649",
        "Laptop": "03642806",
        "Motorbike": "03790512",
        "Mug": "03797390",
        "Pistol": "03948459",
        "Rocket": "04099429",
        "Skateboard": "04225987",
        "Table": "04379243",
        "Grape": "02222222",
        "Real_Grape": "02222221",
    }
    seg_classes = {
        "Airplane": [0, 1, 2, 3],
        "Bag": [4, 5],
        "Cap": [6, 7],
        "Car": [8, 9, 10, 11],
        "Chair": [12, 13, 14, 15],
        "Earphone": [16, 17, 18],
        "Guitar": [19, 20, 21],
        "Knife": [22, 23],
        "Lamp": [24, 25, 26, 27],
        "Laptop": [28, 29],
        "Motorbike": [30, 31, 32, 33, 34, 35],
        "Mug": [36, 37],
        "Pistol": [38, 39, 40],
        "Rocket": [41, 42, 43],
        "Skateboard": [44, 45, 46],
        "Table": [47, 48, 49],
        "Grape": [50,51,52,53,54,55],
        "Real_Grape": [56, 57, 58, 59, 60]
    }
    def __init__(self,root,categories,include_normals,include_colors,split,transform,
                    pre_transform,pre_filter=None,is_test=False):
        super().__init__(root,categories,include_normals,split,transform,
                         pre_transform,pre_filter,is_test)

        #Son class load data
        self.data, self.slices, self.y_mask = self.load_data(
            self.path, include_normals,include_colors,son_load=True)

        # We have perform a slighly optimzation on memory space of no pre-transform was used.
        # c.f self._process_filenames
        if os.path.exists(self.raw_path):
            self.raw_data, self.raw_slices, _ = self.load_data(
                self.raw_path, include_normals,include_colors,son_load=True)
        else:
            self.get_raw_data = self.get
    def load_data(self, path, include_normals,include_colors=False,son_load=False):
        '''This function is used twice to load data for both raw and pre_transformed
        '''
        if not son_load:
            return None,None,None
        data, slices = torch.load(path)

        #data.x = data.x if include_normals and include_colors else None
        #data.x = data.x[:,:3] if include_normals and not include_colors else None
        if include_normals and not include_colors:
            data.x=data.x[:,:3]
        if not include_normals and include_colors:
            data.x=data.x[:,3:]
        if not include_normals and not include_colors:
            data.x=None

        y_mask = torch.zeros(
            (len(self.seg_classes.keys()), 61), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            y_mask[i, labels] = 1

        return data, slices, y_mask

    #def raw_file_names(self):
    #    super(ShapeNetGrape, self).raw_file_names()

    #def processed_raw_paths(self):
    #    super(ShapeNetGrape, self).processed_raw_paths()

    #def processed_file_names(self):
    #    super(ShapeNetGrape, self).processed_file_names()

    def download(self):
        super(ShapeNetGrape, self).download()

    def get_raw_data(self, idx, **kwargs):
        super(ShapeNetGrape, self).get_raw_data()

    def _process_filenames(self, filenames):
        data_raw_list = []
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for name in tq(filenames):
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue
            id_scan += 1
            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:-1]
            y = data[:, -1].type(torch.long)
            category = torch.ones(x.shape[0], dtype=torch.long) * cat_idx[cat]
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            data = Data(pos=pos, x=x, y=y, category=category,
                        id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    #def _save_data_list(self, datas, path_to_datas, save_bool=True):
    #    super(ShapeNetGrape, self)._save_data_list()

    #def _re_index_trainval(self, trainval):
    #    super(ShapeNetGrape, self)._re_index_trainval()

    def process(self):
        super(ShapeNetGrape, self).process()

    def __repr__(self):
        super(ShapeNetGrape, self).__repr__()
class ShapeNetGrapeDataset(ShapeNetDataset):
    def __init__(self, dataset_opt):
        BaseDataset.__init__(self,dataset_opt)
        try:
            self._category = dataset_opt.category
            is_test = dataset_opt.get("is_test", False)
        except KeyError:
            self._category = None
        #print(f'self._data_path:{self._data_path}')
        self._data_path=self._data_path.replace("grape","")
        self.train_dataset = ShapeNetGrape(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            include_colors=dataset_opt.color,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )

        self.val_dataset = ShapeNetGrape(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            include_colors=dataset_opt.color,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            is_test=is_test,
        )

        self.test_dataset = ShapeNetGrape(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            include_colors=dataset_opt.color,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = ShapeNetGrape.seg_classes[key]
        return classes_to_segment
def myinit_dataset(dataset_config) -> BaseDataset:
    try:
        dataset_config.dataroot = hydra.utils.to_absolute_path(
            dataset_config.dataroot)
    except Exception:
        log.error("This should happen only during testing")
    dataset = ShapeNetGrapeDataset(dataset_config)
    return dataset

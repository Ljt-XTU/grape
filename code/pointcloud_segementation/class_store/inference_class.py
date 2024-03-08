# Author:ljt
# Time:2023/8/11 8:22
# Illustration:
import os
import copy
import torch
import hydra
import time
import logging
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import wandb
import numpy as np
import pickle
import torch_geometric.data.data as batch_data

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from class_store.dataset_process import myinit_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.visualization import Visualizer

log = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        if not self.has_training:
            resume = False
            self._cfg.training = self._cfg
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        #if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
        #    self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )
        # Create model and datasets
        if not self._checkpoint.is_empty:
            #self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._dataset: BaseDataset = myinit_dataset(self._checkpoint.data_config)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
            log.info(msg='Checkpoint Done!')
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            #self._model.instantiate_optimizers(self._cfg)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        #log.info(self._model)
        log.info(msg='Model Done!')

        #self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        '''
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )'''


    '''
    Set_input arg "data" format:
        data.batch:batch group which points belong to. [0,0,0...,1,1,1,...,2,2,2]
        data.category:class which sample belong to. [0,0,0,....,0](use_category:True)
        data.grid_size:GridSampling grid size 
        data.id_scan:which sample is loading present 
        data.origin_id:order number of the point in original sample
        pos:(x,y,z) coordinate
        ptr:end position of every batch in data
        x:feature
        y:label
    '''
    def inference(self,infer_data):
        self._model.eval()
        if(self._model.__class__.__name__=='RSConvLogicModel'):
            for key in infer_data.keys:
                if (key in ['category', 'id_scan', 'origin_id', 'pos', 'x', 'y']):
                    infer_data[key]=infer_data[key].unsqueeze(0)
        self._model.set_input(infer_data,self._device)
        y=self._model.forward()
        print(f'y shape:{y.shape}\ny:{y}')
        categories=self._dataset._categories
        lables=self._dataset.test_dataset[0].seg_classes[categories[0]]
        print(f'labels:{lables}\nlabel:{infer_data.y}')
        min_lable=min(lables)
        pre_label=torch.argmax(y[:, min_lable:],-1)+min_lable
        print(pre_label)
        label=infer_data.y
        print(torch.sum(pre_label==label)/label.shape[1])
    def forward_in_test(self):
        self._model.eval()
        test_loader = self._dataset.test_dataloaders[0]
        test_iter = iter(test_loader)  # No loop iteration
        data_i = next(test_iter)
        id_scan = data_i.id_scan.to('cpu').numpy()[0]
        #print(f'data.id_scan:{id_scan}')
        self._model.set_input(data_i, self._device)
        y = self._model.forward()
        # print(f'y shape:{y.shape}\ny:{y}')
        points_num = y.shape[0]
        categories = self._dataset._categories
        lables = self._dataset.test_dataset[0].seg_classes[categories[0]]
        min_lable = min(lables)
        pre_label = torch.argmax(y[:, min_lable:], -1) + min_lable
        pos = data_i.pos.to('cpu').numpy()
        pre_label = pre_label.to('cpu').numpy() - 50
        truth_label = data_i.y.to('cpu').numpy() - 50
        corr = (pre_label == truth_label).astype(np.float)
        print(f'OAcc:{np.sum(corr) / points_num}')

    def save_result(self,pos,truth_label,pre_label,corr,sign):
        #print(f'pos:{pos}\npre_label:{pre_label.reshape(-1,1)}\ntruth_label:{truth_label.reshape(-1,1)}\n'
        #      f'corr:{corr.reshape(-1,1)}')
        result_for_save = np.hstack((pos, truth_label.reshape(-1,1), \
                                     pre_label.reshape(-1,1), corr.reshape(-1,1)))

        dir_path = '../../../seg_result/KPConvPaper_Foshan/9dim'
        filename = 'npoints(20000)_idscan(' + str(sign) + ')_pre.txt'

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        np.savetxt(dir_path + '/' + filename, result_for_save, fmt='%.6f')

    def inference_in_test(self):
        self._model.eval()
        test_loader=self._dataset.test_dataloaders[0]
        test_iter=iter(test_loader) #No loop iteration
        while True:
            try:
                data_i=next(test_iter)
            except:
                break
            id_scan=data_i.id_scan.to('cpu').numpy()[0]
            #print(f'data.id_scan:{id_scan}\nx:{data_i.x}',end='\t')
            self._model.set_input(data_i,self._device)
            y=self._model.forward()
            #print(f'y shape:{y.shape}\ny:{y}')
            points_num=y.shape[0]
            categories = self._dataset._categories
            lables = self._dataset.test_dataset[0].seg_classes[categories[0]]
            min_lable = min(lables)
            pre_label = torch.argmax(y[:, min_lable:], -1) + min_lable
            pos=data_i.pos.to('cpu').numpy()
            pre_label=pre_label.to('cpu').numpy()-55
            truth_label=data_i.y.to('cpu').numpy()-55
            corr=(pre_label == truth_label).astype(np.float)
            #print(f'pos:{pos}\npre_label:{pre_label}\ntruth_label:{truth_label}\n'
            #      f'corr:{corr}\nOAcc:{np.sum(corr) / points_num}')
            print(f'OAcc:{np.sum(corr) / points_num}\tnum points:{points_num}')
           # print(f'pos[0].shape:{pos[0].shape}\ntruth_label[0].shape:{truth_label[0].shape}\n'
           # f'pre_label shape:{pre_label.shape}\ncorr shape:{corr.shape}')
            self.save_result(pos,truth_label,pre_label,corr,id_scan)


    def data_pack(self,data_path,data_name,include_normals, include_colors):
        raw_data = np.loadtxt(data_path)
        dic_data = {}
        if include_normals and include_colors:
            dic_data['x'] = torch.tensor(raw_data[:, 3:-1], dtype=torch.float)
        if include_normals and not include_colors:
            dic_data['x'] = torch.tensor(raw_data[:, 3:6], dtype=torch.float)
        if not include_normals and include_colors:
            dic_data['x'] = torch.tensor(raw_data[:, 6:-1], dtype=torch.float)
        if not include_normals and not include_colors:
            dic_data['x'] = None
        dic_data['y'] = torch.tensor(raw_data[:, -1], dtype=torch.int64)
        dic_data['pos'] = torch.tensor(raw_data[:, :3], dtype=torch.float)
        dic_data['batch'] = torch.tensor(np.zeros((raw_data.shape[0])), dtype=torch.int64)
        dic_data['category'] = torch.tensor(np.zeros((raw_data.shape[0])), dtype=torch.int64)
        dic_data['ptr'] = torch.tensor([0, raw_data.shape[0]], dtype=torch.int64)
        dic_data['filename'] = data_name[:data_name.rfind('.')]

        ''' ueslessness for inference
        dic_data['origin_id']=torch.tensor(np.linspace(0,raw_data.shape[0]-1,raw_data.shape[0]),dtype=torch.int64)
        dic_data['id_scan']=torch.tensor([0],dtype=torch.int64)
        dic_data['grid_size']=torch.tensor([0.0200],dtype=torch.float)'''
        infer_data = batch_data.Data.from_dict(dic_data)
        return infer_data


    def load_raw_data(self,rawdata_dir):
        raw_data_filename=["grape_0(0)", "grape_102(0)", "grape_102(1)", "grape_12(0)",
                               "grape_17(0)", "grape_21(0)", "grape_26(0)", "grape_30(0)",
                               "grape_30(1)", "grape_35(0)", "grape_4(0)", "grape_44(0)",
                               "grape_49(0)", "grape_53(0)", "grape_58(0)", "grape_62(0)",
                               "grape_62(1)", "grape_67(0)", "grape_71(0)", "grape_76(0)",
                               "grape_80(0)", "grape_80(1)", "grape_85(0)", "grape_85(1)",
                               "grape_85(2)", "grape_9(0)", "grape_9(1)", "grape_94(0)",
                               "grape_94(1)", "grape_99(0)", "grape_99(1)"]

        use_normal=getattr(self._cfg.data, "normal", False)
        use_color= getattr(self._cfg.data, "color", False)

        if use_color and use_normal: feature_sign='_9dim'
        if not use_color and use_normal: feature_sign='_normal'
        if use_color and not use_normal: feature_sign='_color'
        if not use_color and not use_normal:feature_sign='_onlypos'

        pickle_file_path='../../../data/grape_rawdata/grape_rawdata'+feature_sign+'.pickle'
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path,'rb') as grape_rawdata:
                self._raw_data=pickle.load(grape_rawdata)
                print(f'Data Load Done!')
            return
        filenames=os.listdir(rawdata_dir)
        filepathes=[os.path.join(rawdata_dir,filename) for filename in filenames]
        self._raw_data=[]
        for (filename,filepath) in zip(filenames,filepathes):
            infer_data = self.data_pack(filepath,filename,use_normal,use_color)
            if (infer_data.filename in raw_data_filename):
                print(f'raw data filename:{infer_data.filename}')
                self._raw_data.append(infer_data)
        with open(pickle_file_path,'wb') as grape_rawdata:
            pickle.dump(self._raw_data,grape_rawdata)
        print('Raw Data Save Done!')


    def inference_on_rawdata(self):
        self._model.eval()
        for i,data_i in enumerate(self._raw_data):
            print(f'data_i.filename:{data_i.filename}',end='\t')
            if (self._model.__class__.__name__ in['RSConvLogicModel','PointNet2_D']):
                for key in data_i.keys:
                    if (key in ['category', 'id_scan', 'origin_id', 'pos', 'x', 'y']):
                        data_i[key] = data_i[key].unsqueeze(0)
            self._model.set_input(data_i, self._device)
            y = self._model.forward()
            points_num = y.shape[0]
            print(f'points_num:{points_num}',end='\t')
            categories = self._dataset._categories
            lables = self._dataset.test_dataset[0].seg_classes[categories[0]]
            min_lable = min(lables)
            pre_label = torch.argmax(y[:, min_lable:], -1) + min_lable
            pos = data_i.pos.to('cpu').numpy()
            pre_label = pre_label.to('cpu').numpy() - 50
            truth_label = data_i.y.to('cpu').numpy() - 50
            #print(f'\tpre_label:{pre_label}\n\ttruth_label:{truth_label}')
            corr = (pre_label == truth_label[0]).astype(np.float)
            print(f'\tOAcc:{np.sum(corr)/points_num}')

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)


    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", False)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if getattr(self._cfg, "wandb", False):
            return getattr(self._cfg.wandb, "log", False)
        else:
            return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.tensorboard, "log", False)
        else:
            return False


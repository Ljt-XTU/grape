# Author:ljt
# Time:2023/7/24 10:24
# Illustration:
import hydra
import importlib
from torch_points3d.datasets.base_dataset import BaseDataset

def get_dataset_class(dataset_config):
    task = dataset_config.task
    # Find and create associated dataset
    try:
        dataset_config.dataroot = hydra.utils.to_absolute_path(
            dataset_config.dataroot)
    except Exception:
        log.error("This should happen only during testing")
    dataset_class = getattr(dataset_config, "class")
    dataset_paths = dataset_class.split(".")
    module = ".".join(dataset_paths[:-1])   #conecting element in join() with "."
    class_name = dataset_paths[-1]
    dataset_module = ".".join(["torch_points3d.datasets", task, module])
    datasetlib = importlib.import_module(dataset_module)


    print('\n--------------Variable printing--------------')
    print('task:{0}\nabsolute_path:{1}\nclass:{2}\ndataset_paths:{3}\ndataset_module:{4}\nclass_name:{5}'. \
          format(task, dataset_config.dataroot, dataset_class, dataset_paths,dataset_module, class_name))
    print('datasetlib:{0}'.format(datasetlib))
    print('--------------Variable printing--------------\n')

    target_dataset_name = class_name
    #找出对应数据集的类（属于BaseDataset的子类）
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset_cls = cls

    if dataset_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (module, class_name)
        )
    return dataset_cls


def instantiate_dataset(dataset_config) -> BaseDataset:
    """Import the module "data/[module].py".
    In the file, the class called {class_name}() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_cls = get_dataset_class(dataset_config)
    dataset = dataset_cls(dataset_config)
    print(dataset)
    return dataset


class MyTrainer:
    def __init__(self,cfg):
        self._cfg=cfg
        self._initialize_trainer()
    def _initialize_trainer(self):

        self._device = "cuda:0"

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        #声明self._dataset是一个BaseDataset类型并赋值为instantiate_dataset(self._cfg.data)
        self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
        print('Dataset Done!')

        if not self.has_training:
            resume = False
            self._cfg.training = self._cfg
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )


        model_config = getattr(self._cfg.models,self._cfg.model_name, None)
        print(getattr(model_config,'down_conv'))
        self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
        self._model.instantiate_optimizers(self._cfg)
        self._model.set_pretrained_weights()
        if not self._checkpoint.validate(self._dataset.used_properties):
            print(
                "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                    self._dataset.used_properties
                )
            )

        self._checkpoint.dataset_properties = self._dataset.used_properties
        print('Model Done!')

        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        print(self._dataset)
        print('Dataloader Done!')
        # Verify attributes in dataset
        print('train_dataset[0]:{0}\ntrain_dataset[1]:{1}'.\
              format(self._dataset.train_dataset[0],self._dataset.train_dataset[0]))
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )

    def train(self):
        self._is_training = True

        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)

            self._train_epoch(epoch)

            if self.profiling:
                return 0

            if epoch % self.eval_frequency != 0:
                continue

            if self._dataset.has_val_loader:
                self._test_epoch(epoch, "val")

            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

        # Single test evaluation in resume case
        if self._checkpoint.start_epoch > self._cfg.training.epochs:
            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

    def _train_epoch(self, epoch: int):

        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                if i % 10 == 0:
                    with torch.no_grad():
                        self._tracker.track(self._model, data=data, **self.tracker_options)

                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                    color=COLORS.TRAIN_COLOR
                )

                if self._visualizer.is_active:
                    self._visualizer.save_visuals(self._model.get_current_visuals())

                iter_data_time = time.time()

                if self.early_break:
                    break

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)


    @property
    def has_training(self):
        return getattr(self._cfg, "training", False)
    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE"\
               and getattr(self._cfg.training, "precompute_multi_scale", False)
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
    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)
    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)


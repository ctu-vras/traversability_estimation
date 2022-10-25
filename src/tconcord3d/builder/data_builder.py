# -*- coding:utf-8 -*-

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_tta
from dataloader.pc_dataset import get_pc_model_class


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          test_dataloader_config=None,
          ssl_dataloader_config=None,
          grid_size=[480, 360, 32], use_tta=False, train_hypers=None):
    train_data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]

    train_ref = train_dataloader_config["return_ref"]

    val_ref = val_dataloader_config["return_ref"]

    if test_dataloader_config is not None:
        test_data_path = test_dataloader_config["data_path"]
        test_imageset = test_dataloader_config["imageset"]
        test_ref = test_dataloader_config["return_ref"]

    # ssl data path for Semi-Supervised training
    ssl_data_path = None
    if ssl_dataloader_config is not None:
        ssl_data_path = ssl_dataloader_config["data_path"]
        ssl_imageset = ssl_dataloader_config["imageset"]
        ssl_ref = ssl_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc = None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=train_data_path, verbose=True)

    # if we want to train in SSL mode
    if train_hypers and ssl_dataloader_config and train_hypers['ssl']:
        train_pt_dataset = SemKITTI(train_data_path, imageset=train_imageset,
                                    return_ref=train_ref, label_mapping=label_mapping,
                                    train_hypers=train_hypers, ssl_data_path=ssl_data_path)
    else:
        train_pt_dataset = SemKITTI(train_data_path, imageset=train_imageset,
                                    return_ref=train_ref, label_mapping=label_mapping,
                                    train_hypers=train_hypers, ssl_data_path=None)

    val_pt_dataset = SemKITTI(val_data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, train_hypers=train_hypers)
    if test_dataloader_config is not None:
        test_pt_dataset = SemKITTI(test_data_path, imageset=test_imageset,
                                   return_ref=test_ref, label_mapping=label_mapping, train_hypers=train_hypers)

    if ssl_dataloader_config is not None:
        ssl_pt_dataset = SemKITTI(ssl_data_path, imageset=ssl_imageset,
                                  return_ref=ssl_ref, label_mapping=label_mapping, train_hypers=train_hypers)

    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True
    )

    if use_tta:
        val_dataset = get_model_class(dataset_config['dataset_type'])(
            val_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
            rotate_aug=True,
            scale_aug=True,
            return_test=True,
            use_tta=True
        )
        collate_fn_BEV_tmp = collate_fn_BEV_tta
    else:
        val_dataset = get_model_class(dataset_config['dataset_type'])(
            val_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
        )
        collate_fn_BEV_tmp = collate_fn_BEV
    if use_tta:
        if test_dataloader_config is not None:
            test_dataset = get_model_class(dataset_config['dataset_type'])(
                test_pt_dataset,
                grid_size=grid_size,
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                ignore_label=dataset_config["ignore_label"],
                rotate_aug=True,
                scale_aug=True,
                return_test=True,
                use_tta=True
            )
        collate_fn_BEV_tmp = collate_fn_BEV_tta
    else:
        if test_dataloader_config is not None:
            test_dataset = get_model_class(dataset_config['dataset_type'])(
                test_pt_dataset,
                grid_size=grid_size,
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                ignore_label=dataset_config["ignore_label"],
            )
        collate_fn_BEV_tmp = collate_fn_BEV

    if ssl_dataloader_config is not None:
        ssl_dataset = get_model_class(dataset_config['dataset_type'])(
            ssl_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
        )

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV_tmp,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"])
    test_dataset_loader = None
    if test_dataloader_config is not None:
        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=test_dataloader_config["batch_size"],
                                                          collate_fn=collate_fn_BEV_tmp,
                                                          shuffle=test_dataloader_config["shuffle"],
                                                          num_workers=test_dataloader_config["num_workers"])
    ssl_dataset_loader = None
    if ssl_dataloader_config is not None:
        ssl_dataset_loader = torch.utils.data.DataLoader(dataset=ssl_dataset,
                                                         batch_size=ssl_dataloader_config["batch_size"],
                                                         collate_fn=collate_fn_BEV,
                                                         shuffle=ssl_dataloader_config["shuffle"],
                                                         num_workers=ssl_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, ssl_dataset_loader

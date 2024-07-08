
## Training script:  

This directory contains the script used to run the full training, ```run_model.py```  

Run the script with ```python run_model.py --cfgfile $path_to_config --train-metadata $path_to_train_dicts --eval-metadata $path_to_eval_dicts --num-gpus $ngpu --run-name $name_of_run --output-dir $path_to_output.```  

You can test this with the double/single_test.json files in ```/tests/deepdisc/test_data/dc2/``` and the config in ```/configs/solo/solo_swin.py```  You should download the pre-trained weights [here](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_swin_b_in21k/f342979038/model_final_246a82.pkl)


The command line options are explained below  

- cfgfile: The configuration file used to build the model, learning rate optimizer, trainer, and dataloaders.
- train-metadata: The training data as a list of dicts stored in json format.  The dicts should have the "instance detection/segmentation" keys specified in the [detectron2 repo](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)
- eval-metadata: The same as the training metadata, but for the evaluation set.
- num-gpus: The number of gpus used to train the model.  Must be a multiple of the batch size specified in the config
- run-name: A string prefix that will be used to save the outputs of the script such as model weights and loss curves
- output-dir: The directory to save the outputs


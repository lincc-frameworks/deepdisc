
DeepDISC Configs
========================================================================================

DeepDISC utilizes Detectron2 configs to build models and data loaders.  Refer to the Detectron2 docs for details on the config parameters.
In general, DeepDISC will use the default config values provided by projects under Detectron2.  However, some parameters should be changed by the user.
Here, we list some parameters that users may want to change.  Be aware that changes will be dependent on your use case and data, and this is not an all-encompaassing list. 


* model.pixel_mean : 'list', a list of the mean of all images in the dataset for each filter
* model.pixel_std : 'list', a list of the standard deviation of all images in the dataset for each filter
* dataloader.train.mapper :  the Mapper that will take the input training metadata and return the input dictionary formatted for the model.
* dataloader.test.mapper :  the Mapper that will take the input testing (eval) metadata and return the input dictionary formatted for the model.
* model.roi_heads.num_classes : the number of classes for classification
* model.backbone.bottom_up.in_chans : the number of filters, i.e., image channels of the input data. This parameter is only for models that use a transformer backbone 


Config References
-----------------

.. literalinclude:: ../../configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_DC2.py
  :language: python
  :linenos:
  :lines: 7-

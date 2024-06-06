
DeepDISC Configs
========================================================================================

DeepDISC utilizes Detectron2 configs to build models and data loaders.  Refer to the Detectron2 docs for details on the config parameters.
In general, DeepDISC will use the default config values provided by projects under Detectron2.  However, some parameters should be changed by the user.
Here, we list some parameters that users may want to change.  Be aware that changes will be dependent on your use case and data, and this is not an all-encompaassing list. 


Config References
-----------------

.. literalinclude:: ../../configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_DC2.py
  :language: python
  :linenos:
  :lines: 7-

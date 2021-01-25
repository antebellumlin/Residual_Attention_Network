# Residual Attention Network

This repository is to reproduce the architecture described in "residual attention network for image classification", including algorithm of attention module and algorithm of attention residual learning. The dataset we used is CIFAR-10, which can be easily identified in the code. There are 5 files in this repository:

* **Result**: this folder contains the results produced by the models.

* **Main_code_for_running.ipynb**: is the main jupyter notebook to run the model. To run this main jupyter notebook, just use the kernel "restart and run all" after deciding which model to use at 4th cell. 

* **ResultsCompare.ipynb**: visualization of the results in "Results" folder.

* **models.py**: contains the code of models. There are 6 models constructed in this project: ARL92, NAL92, ARL56, NAL56, channel_att56 and inception_attention. ARL means attention residual learning and the NAL means model without attention residual learning. The number 92 and 56 means the depth of the attention module. These four models are constructed to make comparison on effect of ARL algorithm and the effect of the depth. Model channel_att56 switch the sigmoid function with channel attention to observe the effect of it. Model inception_attention switch the residual unit with inception module to see the effect of it.

* **residual_units.py**: includes code of algorithm of attention module, algorithm of attention residual learning and inception module, which are used in models.py.

Reference:

https://arxiv.org/abs/1704.06904 <br>
https://github.com/fwang91/residual-attention-network <br>
https://github.com/qubvel/residual_attention_network <br>
https://github.com/koichiro11/residual-attention-network <br>
https://github.com/hakimnasaoui/Image-Scene-Classification/blob/master/Deep_Res_ception_Model.ipynb <br>
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py <br>
https://keras.io/callbacks/ <br>






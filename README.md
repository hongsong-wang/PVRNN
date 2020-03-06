# PVRNN
PVRNN for human motion prediction for this paper:
https://arxiv.org/abs/1906.06514

download data from
https://github.com/una-dinosauria/human-motion-prediction

The two open source
https://github.com/facebookresearch/QuaterNet

https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics
are also very helpful

Run main.py with the following setting:
--residual: True
--veloc: True
--loss_type: 1
--pos_embed: True

Test well with python3.6

If you are using this code, please cite this paper:

Wang, Hongsong, and Jiashi Feng. "PVRED: A position-velocity recurrent encoder-decoder for human motion prediction." arXiv preprint arXiv:1906.06514 (2019).

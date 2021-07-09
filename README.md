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

Test well with python3.6, Pytorch. Run with one GPU.

The visualization of results are shown in
https://www.youtube.com/watch?v=Gmvk8HsyzOc&t=5s 
and
https://www.youtube.com/watch?v=mECybzziYHM

If you are using this code, please cite this paper:

@article{wang2021pvred,
  title={PVRED: A Position-Velocity Recurrent Encoder-Decoder for Human Motion Prediction},
  author={Wang, Hongsong and Dong, Jian and Cheng, Bin and Feng, Jiashi},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}

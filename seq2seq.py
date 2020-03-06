import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim

from enc_dec import Encoder_Decoder
import torch_utils

class EncodeDecodeModel(object):
    def __init__(self, input_size, hidden_size, num_layer, rnn_unit, out_dropout, std_mask, learning_rate, step_size, gamma,
                 residual=False, cuda=False, veloc=False, loss_type=0, pos_embed=False, pos_embed_dim=96 ):
        super(EncodeDecodeModel, self).__init__()
        self.epoch = 0
        self.model = Encoder_Decoder(input_size, hidden_size, num_layer, rnn_unit, residual=residual, out_dropout=out_dropout,
                                 std_mask=std_mask, veloc=veloc, pos_embed=pos_embed, pos_embed_dim=pos_embed_dim, cuda=cuda)
        self.loss_type = loss_type
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        if cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, step_size=step_size, gamma=gamma)

    def scheduler_step(self):
        self.epoch = self.epoch + 1
        self.model_scheduler.step()

    def get_loss(self, outputs, target):
        if self.loss_type == 0:
            total_loss = self.loss(outputs, target[1:])
            return total_loss
        if self.loss_type == 1:
            total_loss = torch_utils.get_train_expmap_to_quaternion_loss(outputs, target[1:])
            return total_loss

    def train(self, input, target):
        self.model.train()
        outputs_enc, outputs = self.model(input, target)
        total_loss = self.get_loss(outputs, target)

        gradient_clip = 0.1
        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()

        return total_loss.item()

    def eval(self, input, target):
        self.model.eval()
        outputs_enc, outputs = self.model(input, target)
        return outputs

if __name__ == '__main__':
    pass

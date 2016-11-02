"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.my_xman= self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        #TODO: define your model here
        return x.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('../data/%s.train'%dataset, '../data/%s.valid'%dataset, '../data/%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    # build
    print "building lstm..."
    #TODO INITIALIZE LSTM HERE

    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
        # validate
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
    print "done"

    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        
    #np.save(output_file, ouput_probabilities)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--dataset', dest='dataset', type=str, default='smaller')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)

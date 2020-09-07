import torch
import argparse
import torch.optim as optim
from torch import nn
# from .src.utils import *
from .src import train

class hyp_parser:
    def __init__(self):
        self.aligned=False
        self.aonly=False
        self.attn_dropout=0.1
        self.attn_dropout_a=0.0
        self.attn_dropout_v=0.0
        self.attn_mask=True
        self.batch_chunk=1 
        self.batch_size=512
        self.clip=0.8
        self.data_path='data' 
        self.dataset='mosi'
        self.embed_dropout=0.25 
        self.f=''
        self.log_interval=30 
        self.lonly=False
        self.lr=5e-03
        self.model='MulT' 
        self.name='mult' 
        self.nlevels=5
        self.no_cuda=False 
        self.num_epochs=100 
        self.num_heads=4
        self.optim='Adam' 
        self.out_dropout=0.0
        self.relu_dropout=0.1 
        self.res_dropout=0.1 
        self.seed=1111
        self.vonly=True 
        self.when=20

def transform(d):
    t_args =hyp_parser()
    print("in transform")
    print(t_args.seed)
    torch.manual_seed(t_args.seed)
    dataset = str.lower(t_args.dataset.strip())
    valid_partial_mode = t_args.lonly + t_args.vonly + t_args.aonly   

    if valid_partial_mode == 0:
        t_args.lonly = t_args.vonly = t_args.aonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.") 

    use_cuda = False    

    output_dim_dict = {
        'mosi': 1,
        'mosei_senti': 1,
        'iemocap': 8
    }   

    criterion_dict = {
        'iemocap': 'CrossEntropyLoss'
    }   

    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if t_args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(t_args.seed)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            use_cuda = True 

    ####################################################################
    #
    # Load the dataset (aligned or non-aligned)
    #
    ####################################################################    

    # print("Start loading the data....") 

    # train_data = get_data(t_args, dataset, 'train')
    # valid_data = get_data(t_args, dataset, 'valid')
    # test_data = get_data(t_args, dataset, 'test')
       
    # train_loader = DataLoader(train_data, batch_size=t_args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_data, batch_size=t_args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=t_args.batch_size, shuffle=True)   

    # print('Finish loading the data....')
    if not t_args.aligned:
        print("### Note: You are running in unaligned mode.")   

    ####################################################################
    #
    # Hyperparameters
    #
    ####################################################################    

    print(t_args)
    hyp_params = t_args
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = (d,d,d)
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = (10,10,10)
    hyp_params.layers = 1
    hyp_params.use_cuda = True
    hyp_params.dataset = 'mosi'
    hyp_params.when = 20
    hyp_params.batch_chunk = 1
    # hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
    hyp_params.model = str.upper('MULT'.strip())
    hyp_params.output_dim = output_dim_dict.get(dataset, 1)
    hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    

    settings = {
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion}
    return hyp_params, settings

# if __name__ == '__main__':
#     hyp_param = transform()
#     test_loss = train.initiate(hyp_params)


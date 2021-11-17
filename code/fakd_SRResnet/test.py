import os
from tqdm import tqdm
import numpy as np
import torch
import data
from option import args
from models import *
import utility
import PIL
import imageio

def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]

def test():
    model.eval()
    with torch.no_grad():     
        for idx_data, d in enumerate(test_loader):
            for idx_scale, scale in enumerate(args.scale):
                d.dataset.set_scale(idx_scale)
                psnr_list = []
                psnr = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = prepare(lr, hr)
                    fms, sr = model(lr)
                    sr = utility.quantize(sr, args.rgb_range)
                
                    save_list = [sr]
                    psnr += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)
                    if args.save_gt:
                        save_list.extend([lr, hr])
                    postfix = ('SR', 'LR', 'HR')
                    for v, p in zip(save_list, postfix):
                        normalized = v[0].mul(255 / args.rgb_range)
                        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                        imageio.imwrite(('../experiment/fakd_SRResnet/result/results-{}/{}_x{}_{}.png'.format(d.dataset.name,filename[0],scale, p)), tensor_cpu.numpy()) 
                        
                psnr /= len(d)
                psnr_list.append(psnr)
                best_psnr = max(psnr_list)
                
                print('[{} x{}]\tPSNR: {:.3f}'.format(
                        d.dataset.name,
                        scale,
                        best_psnr
                    ))
                

   
    
if __name__ == '__main__':
    loader = data.Data(args)
    test_loader = loader.loader_test
       
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    device = torch.device('cpu' if args.cpu else 'cuda')
    
    if args.model == 'FakdSRRes':
        model = fakd_srres.FakdSRRes(args).to(device)
   
    
    if args.TS == 'T':
        model.load_state_dict_teacher(torch.load(args.ckp_path))
    elif args.TS == 'S':
        model.load_state_dict_student(torch.load(args.ckp_path))
    
    test()

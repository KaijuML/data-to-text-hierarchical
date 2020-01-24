"""This file is nearly word-for-word taken from the folder tools in OpenNMT"""
import pkg_resources
import argparse
import torch
import os


def average_checkpoints(checkpoint_files):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        m = torch.load(checkpoint_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']
        
        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)
                
            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)
    
    return {"vocab": vocab, 'opt': opt, 'optim': None,
            "generator": avg_generator, "model": avg_model}
    
def main():
    parser = argparse.ArgumentParser(description='This script merges checkpoints of the same model')
    parser.add_argument('--folder', dest="folder", help="experiment name")
    parser.add_argument('--steps', dest="steps", nargs="+", help="checkpoints step numbers")
    
    args = parser.parse_args()
    
    expfolder = pkg_resources.resource_filename(__name__, 'experiments')
    model_folder = os.path.join(expfolder, args.folder, 'models')
    
    assert os.path.exists(model_folder), f'{model_folder} is not a valid folder'
    
    checkpoint_files = [os.path.join(model_folder, f'model_step_{step}.pt') for step in args.steps]
    
    avg_cp = average_checkpoints(checkpoint_files)
    torch.save(avg_cp, os.path.join(model_folder, 'avg_model.pt'))
    
    
if __name__ == "__main__":
    main()

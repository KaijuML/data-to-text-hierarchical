import pkg_resources
import argparse
import shutil
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an experiment folder")
    parser.add_argument('--name', dest='name', required=True)
    
    args = parser.parse_args()
    
    folder = pkg_resources.resource_filename(__name__, 'experiments')
    
    if not os.path.exists(folder):
        print("Creating a folder 'experiments/' where all experiments will be stored.")
        os.mkdir(folder)
    
    folder = os.path.join(folder, args.name)
    
    if os.path.exists(folder):
        raise ValueError('An experiment with this name already exists')
        
    os.mkdir(folder)
    os.mkdir(os.path.join(folder, 'data'))
    os.mkdir(os.path.join(folder, 'models'))
    os.mkdir(os.path.join(folder, 'gens'))
    os.mkdir(os.path.join(folder, 'gens', 'test'))
    os.mkdir(os.path.join(folder, 'gens', 'valid'))
    
    print(f'Experiment {args.name} created.')
    

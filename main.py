#----------------------------------------------------------------------------------------------------#
# Lambda event classification model application script.
# NOTE: This script will add a bank with model prediction to an input HIPO file.
#----------------------------------------------------------------------------------------------------#

# Data
import numpy as np
import hipopy.hipopy as hp

# Miscellaneous
import os
import sys #NOTE: ADDED
from tqdm import tqdm
import argparse

# ML
import torch
from torch_geometric.data import Data

# Local
from preprocessing import get_event_table, get_sub_array, preprocess, get_bank_keys
from models import *

def main(
        modelpath = "model.pt",
        filename  = "test.hipo", # Recreate this in your $PWD,
        bank      = "ML::pred",
        dtype     = ["D","I"],
        names     = ["pred","label"],
        group     = 300,
        item      = 0,
    ):
    
    # Make sure you have absolute paths
    modelpath = os.path.abspath(modelpath)
    filename  = os.path.abspath(filename)

    # Load model and connect to device
    model = torch.load(modelpath, weights_only=False)
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Show model
    print(model)

    # Open file
    namesAndTypes = {e:dtype[idx] for idx, e in enumerate(names)}
    file = hp.recreate(filename)
    file.newTree(bank,namesAndTypes,group=group,item=item)
    file.open() # IMPORTANT!  Open AFTER calling newTree, otherwise the banks will not be written!

    # Loop file
    for event_num, event in enumerate(file):
        if event_num ==0: print("event.keys() = ",event.keys())#DEBUGGING

        # Set bank names and entry names to look at
        all_keys            = list(event.keys())
        rec_particle_name       = 'REC::Particle'
        rec_particle_keys       = get_bank_keys(rec_particle_name,all_keys)
        
        # Get REC::Particle bank
        rec_particle_event_table = get_event_table(rec_particle_keys,0,event,dtype=float) if rec_particle_name+'_pid' in event.keys() else []

        # Make sure REC::Particle bank is non-empty
        bankdata = np.array([[0.0],[0.0]]) #BY DEFAULT JUST ADD EMPTY ENTRY FOR CLAS12-ANALYSIS

        if len(rec_particle_event_table)>0:

            # Set entries to use as data/truth
            rec_particle_entry_indices   = [0,1,2,3,9,10,11] # pid, px, py, pz, (vx, vy, vz, vt (Only in MC?), charge), beta, chi2pid, status #TODO: SET THESE OUTSIDE LOOPS
            rec_particle_event_x = get_sub_array(rec_particle_event_table,rec_particle_entry_indices)

            # Preprocess data from REC::Particle instead #NOTE: This also automatically selects subarray
            x = preprocess(rec_particle_event_x)
            x = torch.tensor(x,dtype=torch.float32)
            if torch.abs(x).max()>1.0:
                print("DEBUGGING: torch.where(torch.abs(x)>1.0) = ",torch.where(torch.abs(x)>1.0))
                print("DEBUGGING: torch.abs(x).max()            = ",torch.abs(x).max())
            
            # Define edge index
            num_nodes  = len(x)
            edge_index = torch.tensor([[i,j] for i in range(num_nodes) for j in range(num_nodes)],dtype=torch.long)
            
            # Create PyG graph
            data = Data(x=x, edge_index=edge_index.t().contiguous())

            # Apply model
            with torch.no_grad():
                data = data.to(device)
                out  = model(data.x, data.edge_index, data.batch)
                out  = torch.nn.functional.softmax(out,dim=-1)
                pred = out.argmax(dim=1).item()
                out  = out[0,1].item() #NOTE: GIVE PROBABILITY OF SIGNAL CLASS AT INDEX 1.

                # Set bank data
                bankdata = np.array([[out],[pred]])

        file.update({bank : bankdata})

    # Close file
    file.close()

#---------- Main ----------#
if __name__=="__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Apply a PyTorch binary clasification model to a HIPO file and output results to a new HIPO bank in the same file.')
    parser.add_argument('--modelpath', type=str, default='model.pt',help='Path to PyTorch model')
    parser.add_argument('--hipofile',type=str, default='test.hipo',help='Path to input HIPO file')
    parser.add_argument('--bank', type=str, default='ML::pred',help='HIPO bank name to which to write ML results')
    parser.add_argument('--dtype',type=str, default=["D","I"], nargs=2, help='HIPO bank data types ("D" -> double, "F" -> float etc.)')
    parser.add_argument('--names',type=str, default=["pred","label"], nargs=2, help='HIPO bank row names')
    args = parser.parse_args()

    # Run main
    main(
        modelpath = args.modelpath,
        filename  = args.hipofile,
        bank      = args.bank,
        dtype     = args.dtype,
        names     = args.names,
    )

#----------------------------------------------------------------------------------------------------#
# Secondary vertex track pair classification model application script.
# NOTE: This script will add a bank with model prediction to an input HIPO file.
#----------------------------------------------------------------------------------------------------#

# Data
import numpy as np
import hipopy.hipopy as hp

# Miscellaneous
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import argparse

# ML
import torch
from torch_geometric.data import Data

# Local
from preprocessing import get_event_table, get_trajectories_rec_traj, get_bank_keys
from models import *

def main(
        modelpath = "model.pt",
        filename  = "test.hipo", # Recreate this in your $PWD,
        bank      = "ML::vert2",
        dtype     = ["D","I","I","I"],
        names     = ["pred","label","idx1","idx2"],
        group     = 300,
        item      = 1,
        use_data  = False,
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

    #----------------------------------------------------------------------#
    # Loop file
    for event_num, event in enumerate(file):
        if event_num ==0: print("event.keys() = ",event.keys())#DEBUGGING

        # Set bank names and entry names to look at
        all_keys            = list(event.keys())
        rec_traj_name       = 'REC::Traj'
        rec_traj_keys       = get_bank_keys(rec_traj_name,all_keys)
        
        # Get REC::Traj bank
        rec_traj_event_table = get_event_table(rec_traj_keys,0,event,dtype=float) if rec_traj_name+'_pindex' in event.keys() else []

        # Make sure REC::Traj bank is non-empty
        bankdata = [] #BY DEFAULT JUST ADD EMPTY ENTRY FOR CLAS12-ANALYSIS

        if len(rec_traj_event_table)>0:

            # Preprocess REC::Traj into set of same length trajectory lists
            x = get_trajectories_rec_traj(rec_traj_event_table,max_trajectory_entries=30)
            x = torch.tensor(x,dtype=torch.float32)
            if torch.abs(x).max()>1.0:
                print("DEBUGGING: torch.where(torch.abs(x)>1.0) = ",torch.where(torch.abs(x)>1.0))
                print("DEBUGGING: torch.abs(x).max()            = ",torch.abs(x).max())
            
            # Loop pairs of tracks in x
            for x_idx, el, in enumerate(x):

                # Loop pairs of tracks in x *again*
                for x_idx2, el2 in enumerate(x):

                    # Skip identical entries
                    if x_idx==x_idx2: continue

                    # Get track pair
                    x_pair = np.take(x, [x_idx,x_idx2], axis=0)
            
                    # Define edge index
                    num_nodes  = len(x_pair)
                    edge_index = torch.tensor([[i,j] for i in range(num_nodes) for j in range(num_nodes)],dtype=torch.long)
                    
                    # Create PyG graph
                    data    = Data(x=x_pair, edge_index=edge_index.t().contiguous())
                    data.y  = torch.tensor([1] if y[x_idx,x_idx2]>0 else [0],dtype=torch.long) #NOTE: Add extra dimension here so that training gets target batch dimension right.

                    # Apply model
                    with torch.no_grad():
                        data = data.to(device)
                        out  = model(data.x, data.edge_index, data.batch)
                        out  = torch.nn.functional.softmax(out,dim=-1)
                        pred = out.argmax(dim=1).item()
                        out  = out[0,1].item() #NOTE: GIVE PROBABILITY OF SIGNAL CLASS AT INDEX 1.

                        # Set bank data
                        bankentry = [out,pred,x_idx,x_idx2]
                        bankdata.append(bankentry)
            
            # Convert bankdata to numpy
            bankdata = np.array(bankdata)

        # Add data to file if exists
        file.update({bank : bankdata} if len(bankdata)>0 else {})

    # Close file
    file.close()

#---------- Main ----------#
if __name__=="__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Apply a PyTorch binary clasification model to *each* track per event in a HIPO file and output results to a new HIPO bank in the same file.')
    parser.add_argument('--modelpath', type=str, default='model.pt',help='Path to PyTorch model')
    parser.add_argument('--hipofile',type=str, default='test.hipo',help='Path to input HIPO file')
    parser.add_argument('--bank', type=str, default='ML::vert2',help='HIPO bank name to which to write ML results')
    parser.add_argument('--dtype',type=str, default=["D","I","I","I"], nargs=4, help='HIPO bank data types ("D" -> double, "F" -> float etc.)')
    parser.add_argument('--names',type=str, default=["pred","label","idx1","idx2"], nargs=4, help='HIPO bank row names')
    parser.add_argument('--use_data', action='store_true', help='Use data file configuration (currently no difference)')
    args = parser.parse_args()

    # Run main
    main(
        modelpath = args.modelpath,
        filename  = args.hipofile,
        bank      = args.bank,
        dtype     = args.dtype,
        names     = args.names,
        use_data  = args.use_data,
    )

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from train import valid, make_model_folder
import warnings
from utils import *
from Network import *

import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Federatedparser()
eval_ids = os.listdir(os.path.join(args.wesad_path, "valid"))
eval_data = CustomDataset(pkl_files=[os.path.join(args.wesad_path, "valid", id, id+".pkl") for id in eval_ids])
eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn= lambda x:x)
lossf = nn.CrossEntropyLoss()
net = LSTMModel(3, 4, 1, 2)
net.double()
net.to(DEVICE)

def set_parameters(net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)


def fl_evaluate(server_round:int, parameters: fl.common.NDArrays, config:Dict[str, fl.common.Scalar],)-> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(net, parameters)
    history=valid(net, eval_loader, None, lossf, DEVICE)
    save(net, f"./Models/{args.version}/net.pt")
    print(f"Server-side evaluation loss {history['loss']} / accuracy {history['acc']} / precision {history['precision']} / f1score {history['f1score']}")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }
    
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    fl.server.start_server(strategy=fl.server.strategy.FedAvg(evaluate_fn=fl_evaluate), 
                           config=fl.server.ServerConfig(num_rounds=args.round))
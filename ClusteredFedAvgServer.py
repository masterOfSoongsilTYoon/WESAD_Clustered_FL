from collections import OrderedDict
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace, weighted_loss_avg

import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from train import valid, make_model_folder
import warnings
from utils import *
from Network import *
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Federatedparser()
eval_ids = os.listdir(os.path.join(args.wesad_path, "valid"))
eval_data = CustomDataset(pkl_files=[os.path.join(args.wesad_path, "valid", id, id+".pkl") for id in eval_ids])
eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn= lambda x:x)
lossf = nn.CrossEntropyLoss()
net = LSTMModel(3, 4, 1, 2)
net.double()
net.to(DEVICE)
# pca = PCA(2)
kmeans = KMeans(n_clusters=2)

def cosine_similarity_clustering(X):
    cosine = cosine_similarity(X)
    label = kmeans.fit_predict(cosine)
    return label


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


class ClusteredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy | FitRes] | BaseException]) -> Tuple[Parameters | None | Dict[str, bool | bytes | float | int | str]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            labels = cosine_similarity_clustering(weights_results) 
            print(labels.shape)
            
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
        
        
        
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    fl.server.start_server(strategy=ClusteredFedAvg(evaluate_fn=fl_evaluate), 
                           config=fl.server.ServerConfig(num_rounds=args.round))
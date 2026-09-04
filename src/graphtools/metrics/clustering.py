#Torch data and computations
import torch
from torch_geometric.data import Data

def local_clustering_coeff(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the local clustering coefficient for each node in the graph. Defined as:

    C_i = 2T_i / (k_i(k_i - 1))

    T_i = Number of triangles that include node i
    k_i = Degree of node i

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Local clustering coefficient for each node.
    """
    if graph.num_nodes < 2: #Nan values for 1 node graphs
        return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)
    
    # Adjacency matrix
    adj_matrix = torch.zeros((graph.num_nodes, graph.num_nodes), device = device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1

    # Compute local clustering coefficient
    triangles = torch.matmul(torch.matmul(adj_matrix, adj_matrix), adj_matrix).diagonal() # Compute number of triangles for each node
    degree = adj_matrix.sum(dim = 1) # Degree of each node
    
    # Avoid division by zero
    degree[degree < 2] = 2
    
    local_clustering = triangles / (degree * (degree - 1)) #Coefficient
    
    return local_clustering

def global_clustering_coeff(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the global clustering coefficient of the graph. Defined as:

    C = T_{closed} / T_{total}

    T_{closed} = Total number of closed triplets in the graph
    T_{total} = Total number of triplets in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    """
    if graph.num_nodes < 3: #Nan values for graphs with less than 3 nodes
        return torch.tensor(torch.nan, dtype = torch.float32, device = device)

    # Adjacency matrix
    adj_matrix = torch.zeros((graph.num_nodes, graph.num_nodes), device = device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1

    # Compute global clustering coefficient
    closed_triplets = torch.matmul(torch.matmul(adj_matrix, adj_matrix), adj_matrix).diagonal().sum() # Total number of closed triplets
    degree = adj_matrix.sum() # Sum of degrees (self 2-edges paths)
    norm = torch.matmul(adj_matrix, adj_matrix).sum() # Total number of 2-edges paths
    total_triplets =  (norm - degree)# Total number of possible triplets

    # Avoid division by zero
    if total_triplets == 0:
        return torch.tensor(0.0, device=device)

    global_clustering = closed_triplets / total_triplets

    return global_clustering
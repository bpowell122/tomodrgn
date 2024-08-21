"""
Sample latent embeddings along the shortest path through the latent space nearest neighbor graph which connects specified anchor points.
"""
import argparse
import os
from heapq import heappush, heappop
from itertools import pairwise

import numpy as np
import torch
import networkx as nx

from tomodrgn import utils

log = utils.log


def add_args(parser):
    parser.add_argument('z', type=os.path.abspath,
                        help='Input latent embeddings z.pkl file')
    parser.add_argument('--anchors', type=int, nargs='+', required=True,
                        help='Indices of anchor points along desired trajectory. At least 2 points must be specified.')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True,
                        help='Directory in which to store output .txt/.pkl files of path indices and latent embeddings')
    parser.add_argument('--max-neighbors', type=int, default=10,
                        help='The maximum number of neighbors to initially calculate distances for from each latent embedding')
    parser.add_argument('--avg-neighbors', type=float, default=5,
                        help='Used to set a cutoff distance defining connected neighbors such that each embedding will have this many connected neighbors on average')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of particles to calculate distances for at a time')

    return parser


class LatentGraph(object):
    """
    Class for describing connected latent embeddings as a graph based on proximity in latent space.
    """

    def __init__(self,
                 edges: list[tuple[int, int, float]]):
        """
        Initialize a LatentGraph object from a list of connected edges.
        :param edges: list of tuples (src, dest, distance)
        """
        self.nodes = set([x[0] for x in edges] + [x[1] for x in edges])
        self.edges = {x: set() for x in self.nodes}
        self.edge_length = {}
        for s, d, L in edges:
            assert type(s) is int and type(d) is int and type(L) is float
            self.edges[s].add(d)
            self.edge_length[(s, d)] = L

    @classmethod
    def construct_from_array(cls,
                             data: torch.Tensor,
                             batchsize: int,
                             max_neighbors: int,
                             avg_neighbors: int, ):
        """
        Constructor method to create a graph of connected latent embeddings from an array of all latent embeddings.
        :param data: array of latent embeddings, shape (nptcls, zdim)
        :param batchsize: number of particles to calculate distances for at a time
        :param max_neighbors: maximum number of neighbors to initially calculate distances for from each latent embedding
        :param avg_neighbors: used to set a cutoff distance defining connected neighbors such that each embedding will have this many connected neighbors on average
        :return: LatentGraph instance
        """
        nptcls, zdim = data.shape
        # calculate the closest `args.max_neighbors` particles' latent embeddings to each latent embedding
        log('Calculating distances to particles nearest neighbor in latent space')
        n2 = (data * data).sum(-1, keepdim=True)
        ndist = torch.empty(nptcls, max_neighbors, device=data.device)  # squared distance from each latent to `args.max_neighbors` nearest
        neighbors = torch.empty(nptcls, max_neighbors, dtype=torch.long, device=data.device)  # indices of `args.max_neighbors` nearest latents
        for i in range(0, nptcls, batchsize):
            idx_start = i
            idx_stop = i + batchsize
            # (a-b)^2 = a^2 + b^2 - 2ab
            print(f"Working on particles {idx_start}-{idx_stop}")
            batch_dist = n2[idx_start:idx_stop] + n2.t() - 2 * torch.mm(data[idx_start:idx_stop], data.t())
            ndist[idx_start:idx_stop], neighbors[idx_start:idx_stop] = batch_dist.topk(max_neighbors, dim=-1, largest=False)
        # convert d^2 to d
        ndist = ndist.clamp(min=0).pow(0.5)

        # calculate the maximum allowable distance to enforce an average of args.avg_neighbors neighbors per particle
        if avg_neighbors:
            total_neighbors = int(nptcls * avg_neighbors)
            max_dist = ndist.view(-1).topk(total_neighbors, largest=False)[0][-1]
        else:
            max_dist = None

        log(f'Constructing graph of neighbor particles within distance {max_dist} (to enforce average of {avg_neighbors} neighbors)')
        max_dist = max_dist.to("cpu")
        neighbors = neighbors.to("cpu")
        ndist = ndist.to("cpu")
        edges = []
        for i in range(nptcls):
            for j in range(max_neighbors):
                if max_dist is None or ndist[i, j] < max_dist:
                    # edges are defined as (idx_particle, idx_neighbor, dist_to_neighbor)
                    edges.append((int(i), int(neighbors[i, j]), float(ndist[i, j])))

        return LatentGraph(edges)

    def find_path_dijkstra(self,
                           src: int,
                           dest: int) -> tuple[list[int], float] | tuple[None, None]:
        """
        Standard implementation of Dijkstra's algorithm to find the shortest path through a weighted graph.
        Earliest reference I can find for this code is: https://github.com/theannielin/drkung143/blob/master/q9/dijkstra.py
        :param src: index of starting node
        :param dest: index of ending node
        :return: list of node indices connecting src and dest nodes and total distance of that path, or (None, None) if no path can be found
        """
        visited = set()
        unvisited = []
        distances = {}
        predecessors = {}

        distances[src] = 0
        heappush(unvisited, (0, src))

        while unvisited:
            # visit the neighbors
            dist, v = heappop(unvisited)
            if v in visited or v not in self.edges:
                continue
            visited.add(v)
            if v == dest:
                # We build the shortest path and display it
                path = []
                pred = v
                while pred is not None:
                    path.append(pred)
                    pred = predecessors.get(pred, None)
                return path[::-1], dist

            neighbors = list(self.edges[v])

            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    new_distance = dist + self.edge_length[(v, neighbor)]
                    if new_distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_distance
                        heappush(unvisited, (new_distance, neighbor))
                        predecessors[neighbor] = v

        # couldn't find a path
        return None, None

    def plot_graph(self):
        # TODO method to plot graph in UMAP if zdim > 2 (RASTERIZE or else will explode from all graph edges)
        raise NotImplementedError

    def plot_path(self, full_path, anchors):
        # TODO method to plot path in UMAP if zdim > 2
        raise NotImplementedError


def main(args):
    # log args, create output directory
    log(args)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load the latent embeddings array
    data_np = utils.load_pkl(args.z)
    data = torch.from_numpy(data_np)

    # sanity check inputs
    nptcls, zdim = data.shape
    for i in args.anchors:
        assert i < nptcls, f'A particle index in --anchors exceeds the number of particles found in {args.z}: {nptcls}'
    assert len(args.anchors) >= 2, 'At least 2 anchors required to initialize path search'

    # set the device
    device = utils.get_default_device()
    torch.set_grad_enabled(False)
    data = data.to(device)

    # construct the graph of connected neighbors in latent space (connected meaning Euclidean nearest within a threshold)
    graph = LatentGraph.construct_from_array(data=data,
                                             batchsize=args.batch_size,
                                             max_neighbors=args.max_neighbors,
                                             avg_neighbors=args.avg_neighbors)

    # find the shortest path connecting all anchor points
    full_path = []
    for src, dest in pairwise(args.anchors):
        # find the shortest path connecting each sequential pair of anchors
        log(f'Searching for shortest path between anchor points {src} and {dest}')
        path, path_distance = graph.find_path_dijkstra(src, dest)
        dd = data[path].cpu().numpy()  # shape (num_ptcls_path, zdim)
        dists = ((dd[1:, :] - dd[0:-1, :]) ** 2).sum(axis=1) ** .5

        # report details about the found path
        if path is not None:
            log(f'Found shortest path: {" ".join(str(ind) for ind in path)}')
            log(f"Total path distance: {path_distance}")
            log(f'Distances between each neighbor along path: {" ".join(str(dist) for dist in dists)}')
            log(f'Direct distance between source and destination anchor points: {((dd[0] - dd[-1]) ** 2).sum() ** .5}')
        else:
            log("Could not find path!")

        # extend the overall path between all anchor points
        if path is not None:
            if full_path and full_path[-1] == path[0]:
                # avoid repeating the same node twice
                full_path.extend(path[1:])
            else:
                full_path.extend(path)

    # save outputs
    np.savetxt(fname=os.path.join(args.outdir, 'path_particle_indices.txt'), X=full_path)
    utils.save_pkl(data=data_np[full_path], out_pkl=os.path.join(args.outdir, 'path_particle_embeddings.pkl'), )


if __name__ == '__main__':
    main(add_args(argparse.ArgumentParser(description=__doc__)).parse_args())

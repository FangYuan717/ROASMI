import csv
import array
import logging
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it

from copy import deepcopy
from scipy.stats import rankdata
from collections import OrderedDict
from scipy.special import logsumexp
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize

from args import IdentifyArgs

root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)
    
logging.basicConfig(filename='F:/example.log',format='%(name)s: %(message)s', level=logging.DEBUG) #WARNING
LOGGER = logging.getLogger("FactorGraph")

def sigmoid(x):
    s=1/(1 + np.exp(-x))
    return s

def prepare_candidates_data(data_path,normalize):
    
    df = pd.read_csv(data_path)
    lastIndex = df['peak'][0]
    curIndex = df['peak'][0]
    measure_RT = []
    cand_structureList = []
    MS_scoreList = []
    retention_scoreList = []
    dataDict = {}
    candidates = OrderedDict()
    for i in range(len(df)):
        curIndex = df['peak'][i]
        if curIndex != lastIndex:
            dataDict['measure_RT'] = measure_RT
            dataDict['cand_structure'] = np.array(cand_structureList)
            dataDict['MS_score'] = np.array(MS_scoreList)
            dataDict['retention_score'] = np.array(retention_scoreList)
            candidates[lastIndex] = deepcopy(dataDict)
            cand_structureList = []
            MS_scoreList = []
            retention_scoreList = []
            dataDict.clear()
            measure_RT = df['measure_RT'][i]
            cand_structureList.append(df['cand_structure'][i])
            MS_scoreList.append(df['MS_score'][i])
            retention_scoreList.append(df['retention_score'][i])
            lastIndex = curIndex
            continue
        measure_RT = df['measure_RT'][i]
        cand_structureList.append(df['cand_structure'][i])
        dataDict['n_cand'] = len(cand_structureList)
        MS_scoreList.append(df['MS_score'][i])
        retention_scoreList.append(df['retention_score'][i])

    dataDict['measure_RT'] = measure_RT
    dataDict['cand_structure'] = np.array(cand_structureList)
    dataDict['MS_score'] = np.array(MS_scoreList)
    dataDict['retention_score'] = np.array(retention_scoreList)
    dataDict['n_cand'] = len(dataDict['cand_structure'])
    
    candidates[lastIndex] = deepcopy(dataDict)

    # Determine the constant 'c' added as regularizer to the IOKR scores to avoid zero probabilities
    # Collect all MS-scores from the candidates sets
    scores =  np.hstack([cnd["MS_score"] for cnd in candidates.values()])
    print("Minimum MS-score: %.8f" % np.min(scores))
    # Calculate the constant to make all scores >= 0
    if np.any(scores < 0):
        c1 = np.abs(np.min(scores))
        scores += c1
    else:
        c1 = 0.0
    # The regularization constant is 10-times smaller than the overall minimum of scores larger zero.
    c2 = np.min(scores[scores > 0]) / 10
    print("Regularization constant: c1=%.8f, c2=%.8f" % (c1, c2))
    # Regularize the MS-scores, normalise and logarithmise them
    for i in candidates:
        candidates[i]["MS_score"] = np.maximum(c2, candidates[i]["MS_score"] + c1)
        candidates[i]["MS_score"] /= np.max(candidates[i]["MS_score"])
        assert (np.all(candidates[i]["MS_score"] > 0))

        # Probabilities must sum to one
        if normalize:
            candidates[i]["MS_score"] /= np.sum(candidates[i]["MS_score"])

        # Calculate the log-probabilities
        candidates[i]["log_MS_score"] = np.log(candidates[i]["MS_score"])
        
    return candidates
    

class FactorGraph(object):
    def __init__(self, candidates, fac_rt,order_probs, D, norm_order_scores=False):

        self.candidates = candidates
        self.norm_order_scores = norm_order_scores

        self.D_ms = 1.0 - D
        self.D_rt = D

        # Get variables and factors for the MS2 scores
        self.var = list(candidates.keys())  # i = 1, 2, 3, ..., N
        self.fac_ms = [(i, i) for i in self.var]  # (1, 1), (2, 2), ..., (N, N)
        LOGGER.debug("MS-factors: %s" % str(self.fac_ms))

        # Set up factors and order probabilities using retention time information
        self.fac_rt = fac_rt

        # Calculate the order probs given the candidates, the required rt-factors and the transformation function.
        self.order_probs = self._precalculate_order_probs(self.candidates, self.fac_rt, self.norm_order_scores)
        
        LOGGER.debug("RT-factors: %s" % str(self.fac_rt))

        for i, j in self.fac_rt:
            if i not in self.order_probs or j not in self.order_probs[i]:
                raise Exception("'order_probs' is missing the pair (i, j) = (%d, %d)." % (i, j))

        # Variables related to the Sum-Product algorithm: Marginals
        self.R_sum = None
        self.Q_sum = None

        # Variables related to the Max-Product algorithm: Maximum a-Posteriori
        self.R_max = None
        self.Q_max = None
        self.Par_max = None
        self.acc_max = None

    def _log_likelihood(self, Z) -> float:

        llh_ms = 0.0
        llh_rt = 0.0

        # MS-score
        for i, _ in self.fac_ms:
            llh_ms += self.candidates[i]["log_MS_score"][Z[i]]

        # RT-score
        for i, j in self.fac_rt:
            llh_rt += self.order_probs[i][j]["log_retention_score"][Z[i]][Z[j]]

        return self.D_ms * llh_ms + self.D_rt * llh_rt

    def likelihood(self, Z, log=False) -> float:

        if log:
            val = self._log_likelihood(Z)
        else:
            val = np.exp(self._log_likelihood(Z))

        return val

    def get_candidate_list_graph(self) -> nx.Graph:

        G = nx.Graph()

        # Add nodes
        for i in self.var:
            G.add_node(i, retention_time=self.candidates[i]["measure_RT"], n_cand=self.candidates[i]["n_cand"])

        # Add edges
        for i, j in self.fac_rt:
            n_cand_i, n_cand_j = self.order_probs[i][j]["retention_score"].shape

            fp = self.order_probs[i][j]["retention_score"] < 0.5
            tie = self.order_probs[i][j]["retention_score"] == 0.5

            # Get number of molecule-pairs which do not obey the observed retention order (all)
            n_fp = np.sum(fp)
            n_tie = np.sum(tie) / 2.

            # Get number of molecule-pairs which do not obey the observed retention order (top-20)
            # Note: This assumes that the candidates are ordered by their MS2 score in descendent order
            min_i = np.minimum(n_cand_i, 20)
            min_j = np.minimum(n_cand_j, 20)

            n_fp_20 = np.sum(fp[:min_i, :min_j])
            n_tie_20 = np.sum(tie[:min_i, :min_j]) / 2.

            G.add_edge(i, j, weight_all=(n_fp + n_tie) / (n_cand_i * n_cand_j),weight_20=(n_fp_20 + n_tie_20) / (min_i * min_j))

        return G

    @staticmethod
    def _precalculate_order_probs(candidates, fac_rt, norm_scores=False):

        order_probs = OrderedDict()

        for i, j in fac_rt:
            pref_i, pref_j = candidates[i]["retention_score"][:, np.newaxis], candidates[j]["retention_score"][np.newaxis, :]
            
            if i not in order_probs:
                order_probs[i] = OrderedDict()

            t_i, t_j = candidates[i]["measure_RT"], candidates[j]["measure_RT"]

            order_probs[i][j] = {"retention_score": sigmoid(pref_i - pref_j)}

            if norm_scores:
                # normalize "transition" probabilities, such that the sum of probabilities to get from i -> j (for fixed r and all s) is one.
                order_probs[i][j]["retention_score"] = normalize(order_probs[i][j]["retention_score"], norm="l1", axis=1)
                assert (np.allclose(np.sum(order_probs[i][j]["retention_score"], axis=1), 1.0))

            order_probs[i][j]["log_retention_score"] = np.log(order_probs[i][j]["retention_score"])
        
        return order_probs


    @staticmethod
    def _get_normalization_constant_Z(marginals, margin_type):

        if isinstance(marginals, dict) or isinstance(marginals, OrderedDict):
            marg_0 = marginals[0]  # we can choose _any_ if the un-normalized marginals to calculate Z
        elif isinstance(marginals, np.ndarray):
            marg_0 = marginals
        else:
            raise ValueError("Marginal(s) must be of type 'dict', 'OrderedDict' or 'ndarray'.")

        if margin_type == "sum":
            Z = np.sum(marg_0)
        elif margin_type == "max":
            Z = np.max(marg_0)
        else:
            raise ValueError("Invalid margin type '%s'. Choices are 'sum' and 'max'")

        return Z

    @staticmethod
    def _normalize_marginals(marginals, margin_type, normalize):

        if normalize:
            Z = FactorGraph._get_normalization_constant_Z(marginals, margin_type)
            for i in marginals:
                marginals[i] /= Z
        else:
            pass

        return marginals

    
class TreeFactorGraph(FactorGraph):
    def __init__(self, candidates, var_conn_graph, order_probs, D, norm_order_scores=False):
        
        self.var_conn_graph = var_conn_graph

        # Dictionary enabling access to the nodes by their degree
        self.degree_for_var = self.var_conn_graph.degree()
        self.var_for_degree = self._invert_degree_dict(self.degree_for_var)
        self.max_degree = max(self.var_for_degree)  # maximum variable node degree

        # Choose a maximum degree node as root. Ties are broken by choosing the first variable in the variable list.
        self.root = self.var_for_degree[self.max_degree][0]
        LOGGER.debug("Root: %s" % self.root)

        # Create forward- and backward-pass directed trees
        self.di_var_conn_graph_backward = nx.traversal.bfs_tree(self.var_conn_graph, self.root)
        self.di_var_conn_graph_forward = self.di_var_conn_graph_backward.reverse()

        LOGGER.debug("Forward-graph node order: %s" % self.di_var_conn_graph_forward.nodes())
        LOGGER.debug("Backward-graph node order: %s" % self.di_var_conn_graph_backward.nodes())
        LOGGER.debug("%s" % nx.traversal.breadth_first_search.deque(self.di_var_conn_graph_backward))

        # Dictionary enabling access to the neighbors of the variable nodes
        self.var_neighbors = {i: list(neighbors.keys()) for i, neighbors in self.var_conn_graph.adjacency()}

        # Get rt-factors from the MST
        fac_rt = [(src, trg) for (trg, src) in nx.algorithms.traversal.bfs_edges(self.var_conn_graph, source=self.root)]

        super(TreeFactorGraph, self).__init__(candidates=candidates, fac_rt=fac_rt,order_probs=order_probs, D=D, norm_order_scores=norm_order_scores)

    def __str__(self):
        """
        Return a description of the tree like Markov random field.
        :return: string, describing the tree like MRF
        """
        deg = np.repeat(list(self.var_for_degree.keys()), [len(v) for v in self.var_for_degree.values()])
        rt_diffs = [self.var_conn_graph.get_edge_data(u, v)["rt_diff"] for u, v in self.var_conn_graph.edges]

        return "Root: %d\n" \
               "Degree stats: min=%d, max=%d, avg=%.2f, med=%.2f\n" \
               "Retention time differences: min=%.3f, max=%.3f, avg=%.3f, med=%.3f" % \
               (self.root, min(self.var_for_degree), max(self.var_for_degree), np.mean(deg).item(),
                np.median(deg).item(), min(rt_diffs), max(rt_diffs), np.mean(rt_diffs).item(),
                np.median(rt_diffs).item())

    def _forward_pass(self, aggregation_function) -> (np.array, np.array, np.array, nx.Graph):

        R = OrderedDict()
        Q = OrderedDict()

        # FIXME: For 'sum' we assume working in the log-space!
        if aggregation_function == "sum":
            def agg_fun(x):
                return logsumexp(x, axis=1), None  # sum-product algorithm
            backtracking_graph = None
        elif aggregation_function == "max":
            def agg_fun(x):
                max_idc = np.argmax(x, axis=1)
                max_val = x[np.arange(x.shape[0]), max_idc]
                return max_val, max_idc  # max-product algorithm
            backtracking_graph = nx.Graph()

        q_i__ij = None

        for i in nx.algorithms.traversal.dfs_postorder_nodes(self.var_conn_graph, source=self.root):
            j_src = list(self.di_var_conn_graph_forward.predecessors(i))
            j_trg = list(self.di_var_conn_graph_forward.successors(i))
            LOGGER.debug("Forward: var=%d" % i)
            LOGGER.debug("\tsrc=%s, trg=%s" % (str(j_src), str(j_trg)))

            # Initialize the MS-factor node (R-message)
            R[(i, i)] = {i: self.D_ms * self.candidates[i]["log_MS_score"]}
            LOGGER.debug("\tMS-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                np.min(R[(i, i)][i]), np.max(R[(i, i)][i]), np.mean(R[(i, i)][i]).item(),
                np.median(R[(i, i)][i]).item()))

            # Collect R-messages from neighboring factors to build Q-message
            q_i__ij = R[(i, i)][i]  # ms-factor
            for _j_src in j_src:
                q_i__ij = q_i__ij + R[(_j_src, i)][i]  # rt-factors
            assert (q_i__ij.shape == (self.candidates[i]["n_cand"],))

            if len(j_trg) == 0:
                # No target to send messages anymore. We reached the root.
                assert (i == self.root), "Only the root does not have a further target node."
            else:
                # Still a target to send messages. Have not yet reached the root.
                assert (len(j_trg) == 1), "There should be only one target to send messages to."

                j_trg = j_trg[0]  # outgoing edge
                Q[i] = {(i, j_trg): q_i__ij}

                # gamma_rs's, probabilities of the candidates based on the retention order
                _rt_scores = self.D_rt * self.order_probs[i][j_trg]["log_retention_score"]

                LOGGER.debug("\tRT-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                    np.min(_rt_scores), np.max(_rt_scores), np.mean(_rt_scores).item(), np.median(_rt_scores).item()))

                _tmp = agg_fun(_rt_scores.T + Q[i][(i, j_trg)])
                R[(i, j_trg)] = {j_trg: _tmp[0]}

                LOGGER.debug("\tR-message-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                    np.min(R[(i, j_trg)][j_trg]), np.max(R[(i, j_trg)][j_trg]), np.mean(R[(i, j_trg)][j_trg]).item(),
                    np.median(R[(i, j_trg)][j_trg]).item()))

                if aggregation_function == "max":
                    backtracking_graph.add_edge(j_trg, i, best_candidate=_tmp[1])

        if aggregation_function == "max":
            acc = q_i__ij
        else:
            acc = None

        return R, Q, agg_fun, acc, backtracking_graph

    def _backward_pass(self, R, Q, agg_fun):

        for i in nx.algorithms.traversal.dfs_preorder_nodes(self.var_conn_graph, source=self.root):
            j_src = list(self.di_var_conn_graph_backward.predecessors(i))
            j_trg = list(self.di_var_conn_graph_backward.successors(i))
            LOGGER.debug("Backward: var=%d, src=%s, trg=%s" % (i, str(j_src), str(j_trg)))

            assert ((len(j_src) == 0 and i == self.root) or (len(j_src) == 1 and i != self.root))

            # Collect R-messages
            # q-messages going to the MS factor
            Q[i] = OrderedDict([((i, i), np.zeros((self.candidates[i]["n_cand"],)))])
            for j in j_trg:
                Q[i][(i, i)] += R[(j, i)][i]
            for j in j_src:
                Q[i][(i, i)] += R[(i, j)][i]

            # q-messages going to the RT factors
            for _j_trg in j_trg:
                Q[i][(i, _j_trg)] = R[(i, i)][i]
                for j in j_trg:
                    if j == _j_trg:  # collect r message except from the node we send the message
                        continue
                    Q[i][(i, _j_trg)] = Q[i][(i, _j_trg)] + R[(j, i)][i]
                for j in j_src:
                    if j == _j_trg:  # collect r message except from the node we send the message
                        continue
                    Q[i][(i, _j_trg)] = Q[i][(i, _j_trg)] + R[(i, j)][i]

            # gamma_rs's, probabilities of the candidates based on the retention order
            for _j_trg in j_trg:
                _tmp, _ = agg_fun(self.D_rt * self.order_probs[_j_trg][i]["log_retention_score"] + Q[i][(i, _j_trg)])
                if (_j_trg, i) not in R:
                    R[(_j_trg, i)] = {_j_trg: _tmp}
                else:
                    R[(_j_trg, i)][_j_trg] = _tmp
                assert (R[(_j_trg, i)][_j_trg].shape == (self.candidates[_j_trg]["n_cand"],))

        return R, Q

    def sum_product(self):

        # Forward pass & Backward pass with 'sum'
        R, Q, agg_fun, _, _ = self._forward_pass("sum")
        self.R_sum, self.Q_sum = self._backward_pass(R, Q, agg_fun)
        return self

    def max_product(self):

        # Forward pass & Backward pass with 'sum'
        R, Q, agg_fun, self.acc_max, self.Par_max = self._forward_pass("max")
        self.R_max, self.Q_max = self._backward_pass(R, Q, agg_fun)
        return self

    def MAP(self):

        assert ((self.acc_max is not None) and
                (self.Par_max is not None)), "Run 'Max-product' first!"

        # Find Z_max via backtracking
        N = len(self.var)
        Z_max = np.full((N,), fill_value=-1, dtype=int)  # i = N
        idx_max = np.argmax(self.acc_max)
        Z_max[self.root] = idx_max
        for j_src, i in nx.algorithms.traversal.dfs_edges(self.Par_max, source=self.root):
            Z_max[i] = self.Par_max.edges[(j_src, i)]["best_candidate"][Z_max[j_src]] 
            LOGGER.debug("edge=(%d --> %d), best_candidate=%s, Z_max[%d]=%d" % (
                j_src, i, str(self.Par_max.edges[(j_src, i)]["best_candidate"]), i, Z_max[i]))

        for i in range(N):
            assert (Z_max[i] >= 0), "non-negative candidate indices"
            assert (Z_max[i] < self.candidates[i]["n_cand"])

        # Likelihood of Maximum a posteriori Z_max: p_max
        p_max = self.acc_max[idx_max]
        LOGGER.debug("(Z_max, p_max, p_max (lh-function)): (%s, %f, %f)" % (
            str(Z_max), p_max, self.likelihood(Z_max, log=True)))
        np.testing.assert_allclose(self.likelihood(Z_max, log=True), p_max)

        return Z_max, p_max

    def MAP_only(self):
        
        _, _, _, self.acc_max, self.Par_max = self._forward_pass("max")
        return self.MAP()

    def MAP_only__brute_force(self):

        max_llh = -np.inf
        z_max = None

        for z in it.product(*[range(cands["n_cand"]) for cands in self.candidates.values()]):
            llh = self.likelihood(z, log=True)
            if llh > max_llh:
                max_llh = llh
                z_max = z

        p_max = max_llh

        return list(z_max), p_max

    def _marginals(self, R) -> OrderedDict:

        marginals = OrderedDict()

        for i in self.var:
            # Collect r-messages on incoming edges
            r_i = R[(i, i)][i]

            r_ij = np.zeros_like(r_i)
            for j in self.var_neighbors[i]:
                # Note: We do not "know" which directions the messages have traveled, i.e. how to access them from
                #       the R dictionary.
                if (i, j) in R:
                    r_ij += R[(i, j)][i]
                elif (j, i) in R:
                    r_ij += R[(j, i)][i]
                else:
                    raise Exception("Whoops")

            marginals[i] = np.exp(r_i + r_ij)  # go from log-space to linear-space

        return marginals

    def get_sum_marginals(self, normalize=True) -> dict:

        if self.R_sum is None:
            raise RuntimeError("Run 'sum-product' first!")

        return self._normalize_marginals(self._marginals(self.R_sum), "sum", normalize)

    def get_max_marginals(self, normalize=True) -> dict:

        if self.R_max is None:
            raise RuntimeError("Run 'max-product' first!")

        return self._normalize_marginals(self._marginals(self.R_max), "max", normalize)

    @staticmethod
    def _invert_degree_dict(degs) -> OrderedDict:

        degs_out = OrderedDict()
        for node, deg in degs:
            if deg not in degs_out:
                degs_out[deg] = [node]
            else:
                degs_out[deg].append(node)
        return degs_out


class RandomTreeFactorGraph(TreeFactorGraph):
    def __init__(self, candidates, D,order_probs=None,random_state=None, norm_order_scores=False, RT_interval=30):

        self.rs = check_random_state(random_state)
        self.RT_interval = RT_interval

        super(RandomTreeFactorGraph, self).__init__(
            candidates=candidates, order_probs=order_probs,D=D, norm_order_scores=norm_order_scores,
            var_conn_graph=self._get_random_connectivity(candidates, self.rs, self.RT_interval))

    @staticmethod
    def _get_random_connectivity(candidates, rs, RT_interval):

        # Output graph
        var_conn_graph = nx.Graph()

        # Add variable nodes and edges with random weight
        var = list(candidates.keys())
        for i in var:
            var_conn_graph.add_node(i, measure_RT=candidates[i]["measure_RT"])

        for i, j in it.combinations(var, 2):
            rt_i, rt_j = var_conn_graph.nodes[i]["measure_RT"], var_conn_graph.nodes[j]["measure_RT"]
            rt_diff_ij = rt_j - rt_i
            assert (rt_diff_ij >= 0)

            if rt_diff_ij < RT_interval:
                edge_weight = np.inf  # Such edges will not be chosen in the MST
            else:
                edge_weight = rs.rand()

            var_conn_graph.add_edge(i, j, weight=edge_weight, rt_diff=rt_diff_ij)

        # Get minimum spanning tree
        var_conn_graph = nx.algorithms.tree.minimum_spanning_tree(var_conn_graph)

        return var_conn_graph


def get_marginals(candidates, args:IdentifyArgs, rep, normalize=False, norm_order_scores=False):

    TFG = RandomTreeFactorGraph(candidates, D=args.D, random_state=rep, norm_order_scores=norm_order_scores,RT_interval = args.RT_interval)

    # Find the marginals
    if args.margin_type == "max":
        TFG.max_product()  # Forward-backward algorithm
        marg = TFG.get_max_marginals(normalize=normalize)  # Recover the max-marginals
        Z_max, p_max = TFG.MAP()  # Recover the MAP-estimate
    elif args.margin_type == "sum":
        TFG.sum_product()  # Forward-backward algorithm
        marg = TFG.get_sum_marginals(normalize=normalize)
        Z_max, p_max = None, -1

    return rep, marg, Z_max, p_max



if __name__ == '__main__':
    
    candidates = prepare_candidates_data(IdentifyArgs.identify_path,normalize=False)

    # Run Forward-Backward algorithm on the candidates datasets
    res = Parallel(IdentifyArgs.n_jobs)(delayed(get_marginals)(candidates, args=IdentifyArgs, rep=rep,norm_order_scores=False)
                             for rep in range(IdentifyArgs.n_trees))
                   
    # Average the marginals across trees
    marg_identify = {k: np.zeros(v["n_cand"]) for k, v in candidates.items()}
    for i in marg_identify:
        for r in res:
            marg_identify[i] += r[1][i]
        marg_identify[i] /= IdentifyArgs.n_trees

    topk_n = np.zeros((np.max([cnd["n_cand"] for cnd in candidates.values()]) + 2,))
    for i in candidates:
        # Use ranking based on the marginal scores after MS and RT integration
        _scores = - marg_identify[i]
        # Calculate ranks
        _ranks = np.ceil(rankdata(_scores, method="average") - 1).astype("int")
        print("Rank of peak %d:"%(i))
        print(_ranks)
            
        if IdentifyArgs.index_of_correct_structure is not None:
            # Get the contribution of the correct candidate
            _c = 1.0
            _r = _ranks[IdentifyArgs.index_of_correct_structure[i]]
            # For all candidate with the same score, we update their corresponding ranks with their contribution
            topk_n[_r] += _c        

    if IdentifyArgs.index_of_correct_structure is not None:
        topk_n = np.cumsum(topk_n)
        topk_acc = topk_n / len(candidates) * 100
        print("MS + RT: top1=%d, top3=%d , top10=%d"% (topk_n[0],topk_n[2],topk_n[9]))


    



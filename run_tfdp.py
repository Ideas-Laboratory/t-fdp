import numpy as np
import sys
import os
srcpath = os.path.join(os.path.abspath(os.path.dirname("./source_code/"),))
sys.path.append(srcpath)
from tfdp import tFDP  # NOQA: E402

print("It may take a little time for the Numba just-in-time compilation when the first time to run.")


approx_methods = ["ibFFT_GPU", "ibFFT_CPU", "RVS", "BH", "Exact"]

output_dir = "layout_results/t-FDP_approx/"
all_graphs = ['dwt_72', 'lesmis', 'can_96', 'rajat11', 'jazz', 'visbrazil',
              'grid17', 'mesh3e1', 'netscience', 'dwt_419', 'cluster', 'qh882',
              'price_1000', 'dwt_1005', 'cage8', 'btree9', 'bcsstk09',
              '1138_bus', 'qh1484', 'clusterHiera', 'bcspwr07', 'fidapex6', 'block_2000',
              'sierpinski3d', 'lp_ship04l', 'data', '3elt', 'USPowerGrid',
              'add32', 'ca-GrQc', 'commanche', 'EVA', 'bcsstk33',
              'whitaker3', 'crack', 'fe_4elt2', 'fidapex11', 'bcsstk31', 'bcsstk32', 'finan512', 'luxembourg', 'fe_ocean',
              # 'com-dblp.ungraph',
              # 'com-amazon.ungraph',
              # 'roadNet-PA',
              # 'com-youtube.ungraph',
              # 'roadNet-TX',
              # 'roadNet-CA',
              # 'com-orkut.ungraph',
              # 'com-lj.ungraph'
              # Uncomment these lines after downloading and and preprocessing the oversized graphs from SNAP.
             ]

repetitions = 5
for method in approx_methods:
    tfdp = tFDP(algo=method)
    for graph in all_graphs:
        tfdp.readgraph("./data/" + graph + ".mtx")
        print("graph:", graph)
        tfdp.graphinfo()

        total_t = 0
        for k in range(repetitions):
            posfile = "./layout_results/Other/PMDS/" + \
                graph + ".PMDS_" + str(k)
            init = np.loadtxt(posfile, delimiter=" ")
            res, t = tfdp.optimization()
            total_t += t
            np.savetxt(output_dir + method + "/" + graph + "." + method +
                       "_" + str(k), res, delimiter=' ')
        print("algo :", method, ", avg cost time :", total_t/repetitions, "\n")

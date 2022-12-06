from tfdp import tFDP
import numpy as np
print("Initialize the numba. It may take a little time.")

approx_methods = ["ibFFT_GPU","ibFFT_CPU","RVS","BH", "Exact"]
example_graphs = ["dwt_72", "USPowerGrid", "fidapex11"]
repetitions = 5
for method in approx_methods:
    tfdp = tFDP(algo=method)
    for graph in example_graphs:
        tfdp.readgraph("../data/" + graph + ".mtx")
        print("graph:", graph)
        tfdp.graphinfo()
        total_t = 0
        for k in range(repetitions):
            res, t = tfdp.optimization()
            total_t += t
            np.savetxt("layouts/" + graph + "." + method +
                       "_" + str(k), res, delimiter=' ')
        print("algo :", method, ", avg cost time :", total_t/repetitions, "\n")

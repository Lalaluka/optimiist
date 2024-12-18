from benchmarks.benchmark_logs import benchmark_logs

from optimiist.core import optimiist
# from aim import approximate_inductive_miner_new
import pm4py

for i in range(0,5):
  benchmark_logs(
    [(optimiist, "OptIMIIst"), 
     # (approximate_inductive_miner_new, "AIM"),
     (pm4py.discover_petri_net_inductive, "IMf"),
     (pm4py.discovery.discover_petri_net_ilp, "ILP_Miner")
    ],
    "BPI Challenge 2017", 
    "benchmarks/logs/BPI Challenge 2017.xes.gz", 
    {
      "OptIMIIst": {},
    })
import pm4py
from pm4py import discover_eventually_follows_graph

def get_sublog_statistics(log):
  if len(log) == 0:
    return {}, {}, {}, {}, {}, [], [], []

  dfg, start_activities, end_activities = pm4py.discover_dfg(log)
  # We keep the natural dfg for the base cases and standard IM methdos
  nat_dfg = dfg
  
  efg = discover_eventually_follows_graph(log)

  activities = log["concept:name"].unique()

  ifg = {(a, b): efg.get((a, b), 0) - dfg.get((a, b), 0) for a in activities for b in activities if efg.get((a, b), 0) - dfg.get((a, b), 0)}

  # Fill non existing relationships in efg and dfg with 0
  for a in activities:
    for b in activities:
      if (a,b) not in dfg: dfg[(a,b)] = 0
      if (a,b) not in efg: efg[(a,b)] = 0
      if (a,b) not in ifg: ifg[(a,b)] = 0

  # Create binary dfg
  binary_dfg = {}
  for a in activities:
    for b in activities:
      if dfg[(a,b)] > 0:
        binary_dfg[(a,b)] = 1
      else:
        binary_dfg[(a,b)] = 0
  
  return dfg, nat_dfg, efg, ifg, binary_dfg, start_activities, end_activities, activities
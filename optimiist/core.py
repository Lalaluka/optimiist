from optimiist.base_IM import base_case_AIM, im_findCut
from optimiist.evaluate_cut import evalutate_cut
from optimiist.findCut import findCut_OptIMIIst
from optimiist.log_split import split_log
from optimiist.util import get_sublog_statistics
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.process_tree.utils.generic import fold, reduce_tau_leafs
import pm4py
import pandas as pd

def fallthrough_optimiist(log, empty_traces=0):
  log_stats = dfg, _, efg, _, binary_dfg, start_activities, end_activities, activities = get_sublog_statistics(log)

  C = findCut_OptIMIIst(dfg, efg, binary_dfg, start_activities, end_activities, activities)
  # Append tau loop and skip
  C.append((Operator.XOR, activities, []))
  C.append((Operator.LOOP, activities, []))
  operator, L_1, L_2, empty_traces1, empty_traces2 = None, None, None, None, None
  a_res = 0
  for cut in C:
    log_a, log_b, empty_traces_a, empty_traces_b = split_log(log, cut[0], cut[1], cut[2], empty_traces)
    a = evalutate_cut(cut, log, log_a, log_b, empty_traces_a, empty_traces_b, log_stats)[2]
    if a_res < a:
      a_res = a
      operator, L_1, L_2, empty_traces1, empty_traces2 = cut[0], log_a, log_b, empty_traces_a, empty_traces_b
  
  # Set both log columns to datetime
  if "time:timestamp" in log.columns:
    L_1["time:timestamp"] = pd.to_datetime(L_1["time:timestamp"])
    L_2["time:timestamp"] = pd.to_datetime(L_2["time:timestamp"])

  return ProcessTree(operator=operator, children=[optimiist_recursion(L_1, empty_traces1), optimiist_recursion(L_2, empty_traces2)])

def optimiist_recursion(log, empty_cases = 0): 
    base_case = base_case_AIM(log, empty_cases)
    if base_case:
        return base_case

    operator, c1, c2 = im_findCut(log)
    if not operator:
        return fallthrough_optimiist(log)

    sub_dataframe_1, sub_dataframe_2, empty_1, empty_2 = split_log(log, operator, c1, c2, empty_cases)
    return ProcessTree(operator, children=[optimiist_recursion(sub_dataframe_1, empty_1),
                                           optimiist_recursion(sub_dataframe_2, empty_2)])

def optimiist(log):
  tree = optimiist_recursion(log)
  tree = fold(reduce_tau_leafs(tree))
  petri_net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
  return petri_net, initial_marking, final_marking
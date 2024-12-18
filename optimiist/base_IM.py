from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.util.compression import util as comut
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.cuts.factory import CutFactory
from pm4py.algo.discovery.inductive.variants.instances import IMInstance

def base_case_AIM(log, empty_cases):
    # TODO: Implement BaseCases from AIM

    return None

def im_findCut(log):
    uvcl = comut.get_variants(
        comut.project_univariate(log))
    cut = CutFactory.find_cut(IMDataStructureUVCL(uvcl), IMInstance.IM, parameters={})
    if not cut: return None, None, None
    operator = cut[0].operator
    c1 = list(cut[1][0].dfg.end_activities.keys()) + list(cut[1][0].dfg.start_activities.keys())
    c1 += sum([[a, b] for a, b in list(cut[1][0].dfg.graph.keys())], [])
    c2 = []
    for i in range(1, len(cut[1])):
        c2 += list(cut[1][i].dfg.end_activities.keys()) + list(cut[1][i].dfg.start_activities.keys())
        c2 += sum([[a, b] for a, b in list(cut[1][i].dfg.graph.keys())], [])
    return operator, list(set(c1)), list(set(c2))
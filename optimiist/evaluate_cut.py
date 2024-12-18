from typing import Tuple
from pm4py.objects.process_tree.obj import Operator
import numpy as np
import pm4py

def evalutate_cut(cut, log, log_a, log_b, empty_traces_a, empty_traces_b, log_stats):
  if cut[0] == Operator.SEQUENCE:
    return get_seq_conformance(log_stats[0], cut[1], cut[2], log_a, log_b)
  elif cut[0] == Operator.XOR and len(log_a) > 0 and len(log_b) > 0:
    return get_xor_conformance(log_stats[0], cut[1], cut[2])
  elif cut[0] == Operator.PARALLEL:
    return get_and_conformance(log, log_stats[0], cut[1], cut[2],  log_a, log_b)
  elif cut[0] == Operator.LOOP and len(log_a) > 0 and len(log_b) > 0:
    return get_loop_conformance(log, log_stats[0], cut[1], cut[2], log_a, log_b)
  elif cut[0] == Operator.LOOP and len(log_b) == 0:
    return None, None, get_tau_loop_conformance_AIM(log, cut, log_a, empty_traces_a, empty_traces_b, log_stats)
  elif cut[0] == Operator.XOR and len(log_b) == 0:
    return None, None, get_tau_skip_conformance_AIM(log, empty_traces_a)
  else:
    raise Exception("Invalid cut")
  
def calculate_mae(truth_array, input_array):
    if len(truth_array) != len(input_array):
        raise ValueError("Arrays must be of the same length")
        
    # Convert lists to numpy arrays if they are not already
    truth_array = np.array(truth_array)
    input_array = np.array(input_array)
        
    # Calculate the Mean Absolute Error
    mae = np.mean(np.abs(truth_array - input_array))
    return mae

def f1_score(precision, fitness):
    return 2 * (precision * fitness) / (precision + fitness)

def get_xor_conformance(dfg, partitionA, partitionB) -> Tuple[float, float]:
    crossing_edges = 0
    possible_edges = 0

    for a in partitionA:
        for b in partitionB:
            possible_edges += 2

    for edge in dfg:
        if edge[0] in partitionA and edge[1] in partitionB:
            crossing_edges += 1
        if edge[1] in partitionA and edge[0] in partitionB:
            crossing_edges += 1

    fitness = 1-(crossing_edges / possible_edges)

    precision = 1

    f1 = f1_score(precision, fitness)

    return fitness, precision, f1

def get_seq_conformance(dfg, partitionA, partitionB, log_a, log_b) -> Tuple[float, float]:
    # Fitness
    violating_edges = 0
    all_edges = 0
    for edge in dfg:
        all_edges += dfg[edge]
        if edge[0] in partitionB and edge[1] in partitionA:
            violating_edges += dfg[edge]

    fitness = 1-(violating_edges / all_edges)

    # Precision
    # Get start and end activities of the sublogs
    end_a = pm4py.get_end_activities(log_a)

    start_b = pm4py.get_start_activities(log_b)

    base_probabilities = (1 / len(start_b))

    # Get actual transition probabilities from end to start activities
    actual_probabilities = {}
    for a in end_a:
        total_transitions = 0
        for b in start_b:
            total_transitions += dfg[(a,b)]
            actual_probabilities[(a,b)] = dfg[(a,b)]
        if total_transitions != 0:
            for b in start_b:
                actual_probabilities[(a,b)] = actual_probabilities[(a,b)] / total_transitions
        else:
            # This only happens if a is a true end activity without beeing followed by anything.
            for b in start_b:
                actual_probabilities[(a,b)] = 0    

    base_props = [base_probabilities for a in end_a for b in start_b]
    actual_props = [actual_probabilities[(a,b)] for a in end_a for b in start_b]

    precision = 1 - calculate_mae(base_props, actual_props)

    return fitness, precision, f1_score(precision, fitness)

def get_and_conformance(log, dfg, partitionA, partitionB, log_a, log_b) -> Tuple[float, float]:
    # Fitness
    fitness = 1

    # Precision
    variants_base = pm4py.statistics.variants.log.get.get_variants(log)

    variants_a = pm4py.statistics.variants.log.get.get_variants(log_a)
    variants_b = pm4py.statistics.variants.log.get.get_variants(log_b)

    average_variant_length_a = sum([len(variant) for variant in variants_a]) / len(variants_a)
    average_variant_length_b = sum([len(variant) for variant in variants_b]) / len(variants_b)

    expected_variants = 2**(average_variant_length_a + average_variant_length_b)

    precision = len(variants_base) / expected_variants

    return fitness, precision, f1_score(precision, fitness)

def get_loop_conformance(log, dfg, partitionA, partitionB, loop_a, loop_b) -> Tuple[float, float]:
    # Fitness
    # Logs starting with activities from partitionB or ending with activities from partitionB break the loop so decrease fitness
    start = pm4py.get_start_activities(log)
    end = pm4py.get_end_activities(log)

    start_activities_in_B = 0
    end_activities_in_B = 0
    for element in start:
        if element in partitionB:
            start_activities_in_B += start[element]
    for element in end:
        if element in partitionB:
            end_activities_in_B += end[element]

    start_activities_total = sum(start.values())
    end_activities_total = sum(end.values())

    fitness = 1 - ((start_activities_in_B + end_activities_in_B) / (start_activities_total + end_activities_total))

    # Precision

    # Get start and end activities of the sublogs
    start_a = pm4py.get_start_activities(loop_a)
    end_a = pm4py.get_end_activities(loop_a)

    start_b = pm4py.get_start_activities(loop_b)
    end_b = pm4py.get_end_activities(loop_b)
    
    enter_probability = (1 / len(end_a)) * (1 / len(start_b))
    exit_probability = (1 / len(end_b)) * (1 / len(start_a))

    base_enter_probabilities = [enter_probability for a in end_a for b in start_b]
    base_exit_probabilities = [exit_probability for a in end_b for b in start_a]

    # Get actual transition probabilities from end to start activities
    actual_enter_probabilities = {}
    for a in end_a:
        total_transitions = 0
        for b in start_b:
            if (a,b) in dfg:
                total_transitions += dfg[(a,b)]
                actual_enter_probabilities[(a,b)] = dfg[(a,b)]
        if total_transitions != 0:
            for b in start_b:
                actual_enter_probabilities[(a,b)] = actual_enter_probabilities[(a,b)] / total_transitions
        else:
            # This only happens if a is a true end activity without beeing followed by anything.
            for b in start_b:
                actual_enter_probabilities[(a,b)] = 1

    enter_props = [actual_enter_probabilities[(a,b)] for a in end_a for b in start_b]

    actual_exit_probabilities = {}
    for a in end_b:
        total_transitions = 0
        for b in start_a:
            total_transitions += dfg[(a,b)]
            actual_exit_probabilities[(a,b)] = dfg[(a,b)]
        if total_transitions != 0:
            for b in start_a:
                actual_exit_probabilities[(a,b)] = actual_exit_probabilities[(a,b)] / total_transitions
        else:
            # This only happens if a is a true end activity without beeing followed by anything.
            for b in start_a:
                actual_exit_probabilities[(a,b)] = 0

    exit_props = [actual_exit_probabilities[(a,b)] for a in end_b for b in start_a]

    enter_mae = calculate_mae(base_enter_probabilities, enter_props)
    exit_mae = calculate_mae(base_exit_probabilities, exit_props)

    precision = (1 - (enter_mae + exit_mae) / 2)

    # Return
    return fitness, precision, f1_score(precision, fitness)

def get_tau_skip_conformance_AIM(log, empty_traces_a):
    # TODO: Implement TauSkip Scoring

def get_tau_loop_conformance_AIM(log, cut, log_a, empty_traces_a, empty_traces_b, log_stats):
    # TODO: Implement TauLoop Scoring
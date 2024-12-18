from gurobipy import *
from pm4py.objects.process_tree.obj import Operator

def findCut_OptIMIIst(dfg, efg, binfg, start_activities, end_activities, activities):
  seq = seq_p1, seq_p2, obj_val = sequence_cut_base_model(efg, activities)
  xor = xor_p1, xor_p2, obj_val = xor_cut_base_model(dfg, activities)
  par = par_p1, par_p2, obj_val = parralel_cut_base_model(dfg, activities, start_activities, end_activities)
  loop = loop_p1, loop_p2, obj_val, start_redo, end_redo = loop_cut_base_model(dfg, activities, binfg, start_activities, end_activities)

  return [(Operator.SEQUENCE, seq[0], seq[1]), (Operator.XOR, xor[0], xor[1]), (Operator.PARALLEL, par[0], par[1]), (Operator.LOOP, loop[0], loop[1])]

def gurobi_partitioning_to_partition(x):
  partition_1 = []
  partition_2 = []
  for key in x:
    if x[key].x == 1:
      partition_1.append(key)
    else:
      partition_2.append(key)
  return partition_1, partition_2

def sequence_cut_base_model(graph, activities, verb=0):

  masterProblem = Model("price_cuts")

  masterProblem.setParam('OutputFlag', verb)

  x = {}

  for a in activities:
    # Create variables x if the activity is in cluster 1 of false in partition two
    x[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"x_{a}")

  masterProblem.update()

  # Partition cant be empty
  masterProblem.addConstr(quicksum(x[a] for a in activities) >= 1)
  masterProblem.addConstr(quicksum((1 - x[a]) for a in activities) >= 1)

  # Since we maximize the variabes with 1 will be on the left side of the partition
  masterProblem.setObjective(quicksum((x[a]-x[b]) * graph[(a,b)] for a in activities for b in activities), GRB.MAXIMIZE)

  masterProblem.optimize()

  return *gurobi_partitioning_to_partition(x), masterProblem.getObjective().getValue()

def parralel_cut_base_model(dfg, activities, start_activities, end_activities, verb=0):
  # Base idea: maximize the dfg arcs beeing cut by the partition
  # Additionally reduce the flow balance between the partitions.
  # This helps to not cut in a similar way to sequence or loop cuts

    masterProblem = Model("price_cuts")

    masterProblem.setParam('OutputFlag', verb)
  
    x = {}
    z = {}
  
    for a in activities:
      # Create variables x if the activity is in cluster 1 of false in partition two
      x[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"x_{a}")
      for b in activities:
        # Variable linked with x[a] and x[b] is 1 if they are in different partitions
        z[a,b] = masterProblem.addVar(vtype=GRB.BINARY, name=f"z_{a}_{b}")
  

    # Flow cant be negative (minimizing in the negative numbers is the same as maximizing in the positive direction)
    flow = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="flow")
    flow_s = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="flow")
    flow_e = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="flow")

    f_a_s = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_a_s")
    f_b_s = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_b_s")
    f_a_e = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_a_e")
    f_b_e = masterProblem.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_b_e")


    masterProblem.update()

    # Link the variables x with the z variables
    for a in activities:
      for b in activities:
        masterProblem.addConstr(z[a,b] >= x[a]-x[b]) 
        masterProblem.addConstr(z[a,b] >= x[b]-x[a])
        masterProblem.addConstr(z[a,b] <= x[a]+x[b])
        masterProblem.addConstr(z[a,b] <= 2 - x[a] - x[b])
    
    # Partition cant be empty
    masterProblem.addConstr(quicksum(x[a] for a in activities) >= 1)
    masterProblem.addConstr(quicksum((1 - x[a]) for a in activities) >= 1)
  
    # Set flow to the difference of start and end activities in the partitions a good parrallel cut should have a similar amount of start and end activities in both partitions
    masterProblem.addConstr(f_a_s == quicksum((1 - x[a]) * ((start_activities[a] if a in start_activities else 0)) for a in activities))
    masterProblem.addConstr(f_a_e == quicksum((1 - x[a]) * ((end_activities[a] if a in end_activities else 0)) for a in activities))

    masterProblem.addConstr(f_b_s == quicksum((x[a]) * ((start_activities[a] if a in start_activities else 0)) for a in activities))
    masterProblem.addConstr(f_b_e == quicksum((x[a]) * ((end_activities[a] if a in end_activities else 0)) for a in activities))
    
    masterProblem.addConstr(flow_s >= f_a_s - f_b_s)
    masterProblem.addConstr(flow_s >= f_b_s - f_a_s)

    masterProblem.addConstr(flow_e >= f_a_e - f_b_e)
    masterProblem.addConstr(flow_e >= f_b_e - f_a_e)

    masterProblem.addConstr(flow >= flow_e + flow_s)
    
    # Objective
    masterProblem.setObjective(quicksum(z[a,b] * dfg[a,b] for a in activities for b in activities) - flow, GRB.MAXIMIZE)
  
    masterProblem.optimize()
  
    return *gurobi_partitioning_to_partition(x), masterProblem.getObjective().getValue()

def xor_cut_base_model(graph, activities, verb=0):
  
    masterProblem = Model("price_cuts")

    masterProblem.setParam('OutputFlag', verb)
  
    x = {}
    z = {}
  
    for a in activities:
      # Create variables x if the activity is in cluster 1 or false in partition two
      x[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"x_{a}")
      for b in activities:
        # Variable linked with x[a] and x[b] is 1 if they are in different partitions
        z[a,b] = masterProblem.addVar(vtype=GRB.BINARY, name=f"z_{a}_{b}")

    masterProblem.update()

    # Link the variables x with the z variables
    for a in activities:
      for b in activities:
        masterProblem.addConstr(z[a,b] >= x[a]-x[b]) 
        masterProblem.addConstr(z[a,b] >= x[b]-x[a])

    # Partitions cant be empty
    masterProblem.addConstr(quicksum(x[a] for a in activities) >= 1)
    masterProblem.addConstr(quicksum((1 - x[a]) for a in activities) >= 1)
  
    # Since we maximize the variabes with 1 will be on the left side of the partition
    masterProblem.setObjective(quicksum(z[a,b] * graph[(a,b)] for a in activities for b in activities), GRB.MINIMIZE)
  
    masterProblem.optimize()
  
    return *gurobi_partitioning_to_partition(x), masterProblem.getObjective().getValue()

def loop_cut_base_model(dfg, activities, binary_dfg, start_activities, end_activities, verb=0):
  # Idea A: minimize flow between partitions that is not exit redo to start do and end do to start redo

  masterProblem = Model("price_cuts")

  masterProblem.setParam('OutputFlag', verb)

  # Singleton variables
  x = {} # Partition variable 1 is do partition, 0 is redo partition
  redo_start = {} # Node is a start activity of the redo loop
  redo_end = {} # Node is a end activity of the redo loop

  # Dualton variables
  cross_partition_flow = {} # Arc is between partitions that is not connecting start and end activities of the do and redo partition
  end__redo_start = {} # Arc is between end activities of the do partition and start activities of the redo partition
  redo_end__start = {} # Arc is between end activities of the redo partition and start activities of the do partition

  # Create variables
  for a in activities:
    x[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"x_{a}")
    redo_start[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"redo_start_{a}")
    redo_end[a] = masterProblem.addVar(vtype=GRB.BINARY, name=f"redo_end_{a}")

    for b in activities:
      cross_partition_flow[a,b] = masterProblem.addVar(vtype=GRB.BINARY, name=f"cross_partition_flow_{a}_{b}")
      end__redo_start[a,b] = masterProblem.addVar(vtype=GRB.BINARY, name=f"end__redo_start_{a}_{b}")
      redo_end__start[a,b] = masterProblem.addVar(vtype=GRB.BINARY, name=f"redo_end__start_{a}_{b}")

  masterProblem.update()

  # Partitions cant be empty
  masterProblem.addConstr(quicksum(x[a] for a in activities) >= 1)
  masterProblem.addConstr(quicksum((1 - x[a]) for a in activities) >= 1)

  # redo_start and redo_end are only 1 if the activity is in the redo partition
  for a in activities:
    masterProblem.addConstr(redo_start[a] <= (1 - x[a]))
    masterProblem.addConstr(redo_end[a] <= (1 - x[a]))

  # Redo_start activities are activities that have incoming arcs from end activities of the do partition
  # Redo_end activities are activities that have outgoing arcs to start activities of the do partition
  for a in activities:
    for b in activities:
      if a in end_activities:
        masterProblem.addConstr(redo_start[b] >= (x[a]-x[b]) * binary_dfg[(a,b)])
        
      if b in start_activities:
        masterProblem.addConstr(redo_end[a] >= (x[b]-x[a]) * binary_dfg[(a,b)])

  # End to redo start flow is one if activity a is an end activity in partition 1 and activity b is a redo start activity in partition 2
  # Redo end to start flow is one if activity a is a redo end activity in partition 1 and activity b is a start activity in partition 2
  for a in activities:
    for b in activities:
      # End to redo start
      masterProblem.addConstr(end__redo_start[a,b] >= redo_start[b] + (1 if a in end_activities else 0) - 1)
      masterProblem.addConstr(end__redo_start[a,b] <= redo_start[b])
      masterProblem.addConstr(end__redo_start[a,b] <= (1 if a in end_activities else 0))
      # Redo end to start
      masterProblem.addConstr(redo_end__start[a,b] >= redo_end[a] + (1 if b in start_activities else 0) - 1)
      masterProblem.addConstr(redo_end__start[a,b] <= redo_end[a])
      masterProblem.addConstr(redo_end__start[a,b] <= (1 if b in start_activities else 0))

  # Cross partition flow are arcs that are not between start and end activities of the do and redo partition but between the partitions
  for a in activities:
    for b in activities:
      masterProblem.addConstr(cross_partition_flow[a,b] >= (x[a]-x[b]) * binary_dfg[(a,b)] - end__redo_start[a,b] - redo_end__start[a,b])
      masterProblem.addConstr(cross_partition_flow[a,b] >= (x[b]-x[a]) * binary_dfg[(a,b)] - end__redo_start[a,b] - redo_end__start[a,b])

  masterProblem.update()

  # Sum of properly cut arcs minus the sum of arcs that are not properly cut and the sum of end and start activities that are not in the right partition
  masterProblem.setObjective(quicksum((redo_end__start[a,b] * dfg[a,b] + end__redo_start[a,b] * dfg[a,b]) for a in activities for b in activities) 
                             - quicksum((1 - x[a]) * ((start_activities[a] if a in start_activities else 0) + (end_activities[a] if a in end_activities else 0)) for a in activities)
                             - quicksum(cross_partition_flow[a,b] * dfg[a,b] for a in activities for b in activities), GRB.MAXIMIZE)

  masterProblem.optimize()

  return *gurobi_partitioning_to_partition(x), masterProblem.getObjective().getValue(), *gurobi_extract_redo(redo_start, redo_end)

def gurobi_extract_redo(redo_start, redo_end):
  lredo_start = []
  lredo_end = []
  for key in redo_start:
    if redo_start[key].x == 1:
      lredo_start.append(key)
  for key in redo_end:
    if redo_end[key].x ==1:
      lredo_end.append(key)
  return lredo_start, lredo_end

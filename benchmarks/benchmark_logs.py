import os
import pm4py
import time
import uuid
import random

def benchmark_logs(algorithms: list, name, log_path, algo_params):
  # Create folder for log
  f_uid = str(uuid.uuid4())
  if not os.path.exists("benchmarks/log_benchmarks/" + name + "/_" + f_uid):
    os.makedirs("benchmarks/log_benchmarks/" + name + "/_" + f_uid)

  log = pm4py.read_xes(log_path)
  # Split the log into 80/20 train/test split based on the number of traces
  # Get list of trace ids
  trace_ids = log["case:concept:name"].unique()
  # Shuffle the list of trace ids
  random.shuffle(trace_ids)
  # Split the trace ids
  train_ids = trace_ids[:int(len(trace_ids) * 0.8)]

  # Create train and test logs
  train_log = log[log["case:concept:name"].isin(train_ids)]
  test_log = log[~log["case:concept:name"].isin(train_ids)]

  # Print trace count in train and test logs
  print("Train log trace count:", len(train_log))
  print("Test log trace count:", len(test_log))

  # Save the train and test logs
  pm4py.write_xes(train_log, "benchmarks/log_benchmarks/" + name + "/_" + f_uid + "/train_log.xes")
  pm4py.write_xes(test_log, "benchmarks/log_benchmarks/" + name + "/_" + f_uid + "/test_log.xes")

  # Create results file
  with open(f"benchmarks/log_benchmarks/{name + "/_" + f_uid}/results.csv", "w") as f:
    f.write("algorithm,runtime,fitness,precision,sound,size,log\n")

  for algorithm, alg_name in algorithms:
    start_time = time.process_time()
    net, im, fm = algorithm(train_log, **algo_params.get(alg_name, {}))
    end_time = time.process_time()
    runtime = end_time - start_time

    # Save the model
    pm4py.write_pnml(net, im, fm, f"benchmarks/log_benchmarks/{name + '/_' + f_uid}/{alg_name}.pnml")

    # Get model size (number of transitions, places and arcs)
    size = len(net.transitions) + len(net.places) + len(net.arcs)

    # Conformance check
    is_sound = pm4py.analysis.check_soundness(net, im, fm)

    # Run conformace check
    alignments_fitness = pm4py.conformance.fitness_alignments(test_log, net, im, fm)
    
    alignments_precision = pm4py.conformance.precision_alignments(log, net, im, fm)

    # Write the results to a csv file
    with open(f"benchmarks/log_benchmarks/{name + '/_' + f_uid}/results.csv", "a") as f:
      f.write(f"{alg_name},{runtime},{alignments_fitness['log_fitness']},{alignments_precision},{is_sound[0]},{size},{name}\n")

  # Move the file IMLP-CALLS.txt to the folder
  # os.rename("IMLP-CALLS.txt", f"benchmarks/log_benchmarks/{name + '/_' + f_uid}/IMLP-CALLS.txt")

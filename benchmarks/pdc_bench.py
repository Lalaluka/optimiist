import os
import pm4py
import datetime
import pandas

train_dir = "benchmarks/logs/PDC2024/Training Logs"
base_dir = "benchmarks/logs/PDC2024/Base Logs"
test_dir = "benchmarks/logs/PDC2024/Test Logs"
truth_dir = "benchmarks/logs/PDC2024/Ground Truth Logs"

def pdc_bench(result_dir, algorithm,name):

    result_list = []
    for log_file in os.listdir(train_dir):

        tp, fp, tn, fn = 0, 0, 0, 0
        log = pm4py.read_xes(f"{train_dir}/{log_file}")
        log["time:timestamp"] = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(0, log.shape[0])]
        net, im, fm = algorithm(log)
        # net, im, fm = pm4py.convert_to_petri_net(tree)
        g = int(log_file[14])
        h = int(log_file[15])
        log_file = log_file[:14] + ".xes"
        base = pm4py.read_xes(f"{base_dir}/{log_file}")
        base["time:timestamp"] = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(0, base.shape[0])]
        test = pm4py.read_xes(f"{test_dir}/{log_file}")
        test["time:timestamp"] = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(0, test.shape[0])]
        real = pm4py.read_xes(f"{truth_dir}/{log_file}")
        real["time:timestamp"] = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(0, real.shape[0])]

        for case in test["case:concept:name"].unique():
            score_base = pm4py.fitness_alignments(base[base["case:concept:name"] == case], net, im, fm)["log_fitness"]
            score_test = pm4py.fitness_alignments(test[test["case:concept:name"] == case], net, im, fm)["log_fitness"]
            if (score_test > score_base) == real[real["case:concept:name"] == case].iloc[0]["case:pdc:isPos"]:
                if score_test > score_base:
                    tp += 1
                else:
                    tn += 1
            else:
                if score_test > score_base:
                    fp += 1
                else:
                    fn += 1
        code = log_file.split(".")[0].split("_")[1]
        prec = (tp / (tp + fp))
        fit = (tp / (tp + fn))
        try:
            f1 = (2 * prec * fit) / (prec + fit)
        except:
            f1 = None
        result = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "A": int(code[0]), "B": int(code[1]), "C": int(code[2])
            , "D": int(code[3]), "E": int(code[4]), "F": int(code[5]), "G": g, "H": h, "Precision": prec, "Fitness": fit,
                  "F1": f1}
      
        result_list.append(result)
        data = pandas.DataFrame(result_list)
        accuracy = (1.0 * (data["TP"] + data["TN"])) / (data["TP"] + data["TN"] + data["FP"] + data["FN"])
        # print(result)
        # print(accuracy)
        data.to_csv(f"{result_dir}/{name}.csv")

        # print(pandas.DataFrame(result_list)["F1"].mean())

    data = pandas.DataFrame(result_list)
    accuracy = (1.0 * (data["TP"] + data["TN"])) / (data["TP"] + data["TN"] + data["FP"] + data["FN"])
    pprecision = (1.0 * data["TP"]) / (data["TP"] + data["FP"])
    precall = (1.0 * data["TP"]) / (data["TP"] + data["FN"])
    pfscore = (2 * pprecision * precall) / (pprecision + precall)
    nprecision = (1.0 * data["TN"]) / (data["TN"] + data["FN"])
    nrecall = (1.0 * data["TN"]) / (data["TN"] + data["FP"])
    nfscore = (2 * nprecision * nrecall) / (nprecision + nrecall)
    precision = (data["TP"] * pprecision + data["TN"] * nprecision) / (data["TP"] + data["TN"])
    recall = (data["TP"] * precall + data["TN"] * nrecall) / (data["TP"] + data["TN"])
    fscore = (data["TP"] * pfscore + data["TN"] * nfscore) / (data["TP"] + data["TN"])

    data["N"] = nfscore
    data["P"] = pfscore
    data["T"] = fscore
    data.to_csv(f"{result_dir}/{name}.csv")

    average_fscore = data["T"].mean()
    print(f"Average F-score across all logs: {average_fscore}")

    # Also create the F-score for logs having H = 0
    data_h0 = data[data["H"] == 0]
    average_fscore_h0 = data_h0["T"].mean()
    print(f"Average F-score across all logs with H = 0: {average_fscore_h0}")
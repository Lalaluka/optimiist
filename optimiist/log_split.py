from pm4py.objects.process_tree.obj import Operator
import pandas as pd

def split_case_in_loop(log, p1, p2):
  if log.shape[0] == 1: return log
  case_uuid = 0  # Initialize with a UUID
  case_list = []
  for i in range(0, log.shape[0]):
      case_list.append(log.iloc[i]["case:concept:name"] + "🚫️CASESPLITTER🚫️" + str(case_uuid))
      if log.iloc[i]["concept:name"] not in p1 != log.iloc[i - 1]["concept:name"] in p1:
          case_uuid += 1
  log["case:concept:name"] = case_list
  return log
    
def split_loop(log, loop_p1, loop_p2, empty_cases = 0):
  log = log.groupby("case:concept:name", group_keys=False).apply(lambda x: split_case_in_loop(x, loop_p1, loop_p2))
  log1 = log[log["concept:name"].isin(loop_p1)]
  log2 = log[log["concept:name"].isin(loop_p2)]
  return log1, log2, empty_cases, 0

def split_case_in_tau_loop(log, start, end) -> pd.DataFrame:
    if log.shape[0] == 1: return log
    case_uuid = 0  # Initialize with a UUID
    case_list = []
    for i in range(0, log.shape[0]):
        case_list.append(log.iloc[i]["case:concept:name"] + "🚫️CASESPLITTER🚫️" + str(case_uuid))
        if i == 0 and log.iloc[i]["concept:name"] in start and log.iloc[i]["concept:name"] in end:
            case_uuid += 1
        elif i + 1 < log.shape[0]:
          if i > 0 and log.iloc[i]["concept:name"] in end and log.iloc[i + 1]["concept:name"] in start:
            case_uuid += 1
    log["case:concept:name"] = case_list
    return log

def split_tau_loop(log: pd.DataFrame, empty_cases: int) -> tuple[list, list, int, int]:
  start = log.groupby("case:concept:name").first().groupby("concept:name").size()
  end = log.groupby("case:concept:name").last().groupby("concept:name").size()
  log = log.groupby("case:concept:name", group_keys=False).apply(lambda x: split_case_in_tau_loop(x, start, end))
  p2 = pd.DataFrame(columns=log.columns)
  if "time:timestamp" in log.columns:
    log.loc[:, "time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    p2.loc[:, "time:timestamp"] = pd.to_datetime(p2["time:timestamp"])
  return log, p2, empty_cases, 0

def split_log(log, operator, p1, p2, empty_cases) -> tuple[list, list, int, int]:
  if operator == Operator.XOR and p2 == []:
    return log, pd.DataFrame({"concept:name": [], "case:concept:name": [], "time:timestamp": []}), 0, 0
  if operator == Operator.LOOP and p2 == []:
    return split_tau_loop(log, empty_cases)
  if operator == Operator.LOOP and p2 != []:
    return split_loop(log, p1, p2, empty_cases)

  sub_log_1 = log[log["concept:name"].isin(p2)].reset_index(drop=True)[["case:concept:name", "concept:name", "time:timestamp"]]
  sub_log_2 = log[log["concept:name"].isin(p2)].reset_index(drop=True)[["case:concept:name", "concept:name", "time:timestamp"]]
  empty_cases_1 = log["case:concept:name"].nunique() - sub_log_1["case:concept:name"].nunique()
  empty_cases_2 = log["case:concept:name"].nunique() - sub_log_2["case:concept:name"].nunique()
  return sub_log_1, sub_log_2, empty_cases + empty_cases_1, empty_cases + empty_cases_2

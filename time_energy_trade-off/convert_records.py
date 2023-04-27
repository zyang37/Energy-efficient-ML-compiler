import os
import sys
import json
import pickle
import numpy as np
from tvm.autotvm.record import load_from_file
from matplotlib import pyplot as plt

from collections import namedtuple

# class MeasureInput(namedtuple("MeasureInput", ["target", "task", "config"]))
# class MeasureResult(namedtuple("MeasureResult", 
#       ["costs", "error_no", "all_cost", "timestamp", "energy"], defaults={"energy": None}))

def calculate_average_gflops_per_watt(inp, res):
  average_time_s = np.mean(res.costs)
  average_flops = inp.task.flop / average_time_s
  average_energy_per_inference_J = res.energy / len(res.costs)
  average_watts = average_energy_per_inference_J / average_time_s
  average_flops_per_watt = average_flops / average_watts
  average_gflops_per_watt = average_flops_per_watt / 1e9  # convert to gigaflops
  return average_gflops_per_watt, average_energy_per_inference_J



if __name__ == '__main__':
    records_file = sys.argv[1]

    readable_records_root = "readable_records/"
    if not os.path.exists(readable_records_root):
        os.makedirs(readable_records_root)

    output_file = os.path.join(readable_records_root, records_file.replace(".json", ".pkl"))

    records = load_from_file(records_file)

    workloads_dict = {}
    for i, (inp, res) in enumerate(records):
        workload = inp.task.workload
        if workload not in workloads_dict:
            workloads_dict[workload] = {'avg_times':[], 'gfpws':[], 'energyJ':[]}

        try:
            average_gflops_per_watt, average_energy_per_inference_J = calculate_average_gflops_per_watt(inp, res)
        except:
            average_gflops_per_watt = 0

        workloads_dict[workload]['avg_times'].append(np.mean(res.costs))
        workloads_dict[workload]['gfpws'].append(average_gflops_per_watt)
        workloads_dict[workload]['energyJ'].append(average_energy_per_inference_J)

    # save as pickle file
    with open(output_file, "wb") as f:
        pickle.dump(workloads_dict, f)
import numpy as np
from tvm.autotvm.record import load_from_file
import plotly.express as px

records_file = "resnet-18-v2-autotuning_with_energy.json"

records = load_from_file(records_file)

workloads_dict = {}
for inp, res in records:
  workload = inp.task.workload
  if workload not in workloads_dict:
    workloads_dict[workload] = []
  workloads_dict[workload].append((inp, res))


def calculate_average_gflops_per_watt(inp, res):
  average_time_s = np.mean(res.costs)
  average_flops = inp.task.flop / average_time_s
  average_energy_per_inference_J = res.energy / len(res.costs)
  average_watts = average_energy_per_inference_J / average_time_s
  average_flops_per_watt = average_flops / average_watts
  average_gflops_per_watt = average_flops_per_watt / 1e9  # convert to gigaflops

  return average_gflops_per_watt

for i, (workload, records) in enumerate(workloads_dict.items()):
  history = []
  for record in records:
    try:
      average_gflops_per_watt = calculate_average_gflops_per_watt(*record)
    except:
      average_gflops_per_watt = 0
    history.append(average_gflops_per_watt)

  fig = px.line(x=range(len(history)), y=history)
  fig.update_layout(
    title=f"Workload: {i}",
  )

  # save figure in plotting directory with name of the workload
  fig.write_image("plotting/%s.png" % (i))




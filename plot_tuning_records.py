import numpy as np
from tvm.autotvm.record import load_from_file
from matplotlib import pyplot as plt

records_file = "resnet-101-v2-autotuning_with_energy.json"

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

best_records = []
plot_legends = []
fig, ax = plt.subplots(figsize=(10, 7))
for i, (workload, records) in enumerate(workloads_dict.items()):
  history = []
  plot_legends.append(workload[0]+' '+str(i))
  best = -1
  best_record = None

  new_records = []
  for record in records:
    try:
      average_gflops_per_watt = calculate_average_gflops_per_watt(*record)
    except:
      average_gflops_per_watt = 0
    if average_gflops_per_watt > best:
        best = average_gflops_per_watt
        best_record = record
    history.append(best)

    new_records.append((record, average_gflops_per_watt))

  new_records.sort(key=lambda a: a[1], reverse=True)


  ax.plot(range(len(history)), history)
  best_records.append(new_records[:5])

ax.set_title('Best GFlops/watt by iteration')
ax.legend(plot_legends, bbox_to_anchor=(1.14, 0.4), loc=5)
ax.xaxis.set_label_text('Iterations')
ax.yaxis.set_label_text('Average GFlops/Watt')
plt.tight_layout()
plt.savefig('resnet-34.png')

for i, new_records in enumerate(best_records):
    print("Best 5 records for: ", list(workloads_dict.keys())[i][0] + ' ' + str(i))
    for record in new_records:
        print('\t' + str(record[0][0].config._entity_map))

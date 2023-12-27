import matplotlib.pyplot as plt
import numpy as np
import json

result_ARC_path="model\\ARC\\trainer_state.json"
result_CYCIC_path="model\\CYCIC\\trainer_state.json"

with open(result_ARC_path,'r') as f:
    result_ARC=json.load(f)
with open(result_CYCIC_path,'r') as f:
    result_CYCIC=json.load(f)

loss_ARC=[d['loss'] for d in result_ARC['log_history'][:-1]]
plt.cla()
plt.title("Training Loss(ARC)")
plt.xlabel("optimization step")
plt.ylabel("loss")
plt.plot([i for i in range(len(loss_ARC))], loss_ARC, 'r')   # red line without marker
plt.savefig('loss_arc.png')


loss_CYCIC=[d['loss'] for d in result_CYCIC['log_history'][:-1]]
plt.cla()
plt.title("Training Loss(CYCIC")
plt.xlabel("optimization step")
plt.ylabel("loss")
plt.plot([i for i in range(len(loss_CYCIC))], loss_CYCIC, 'r')   # red line without marker
plt.savefig('loss_cycic.png')


# & max\_seq\_length & learning\_rate &batch\_size &total\_optimization\_step
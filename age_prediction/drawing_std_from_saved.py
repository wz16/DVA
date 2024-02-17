import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import pickle

font = {'size'   : 15}

matplotlib.rc('font', **font)

with open('save_dict/heter_fixed_epoch0.pickle', 'rb') as file:
    
    pickle_file = pickle.load(file)
  
    std_0 = pickle_file['std']
    ys = pickle_file['ys_list']
    real_age = pickle_file['real_age']

with open('save_dict/heter_fixed_epoch20.pickle', 'rb') as file:
    loaded_dict = pickle.load(file)
    s2yroot = loaded_dict["s2yroot"]
    y = loaded_dict["y"]
    s3yroot = loaded_dict["s3yroot"]
    std = loaded_dict["std"]
    y_noise = loaded_dict["y_noise"]

with open('save_dict/heter_heter_epoch20.pickle', 'rb') as file:
    loaded_dict2 = pickle.load(file)
    s3yroot2 = loaded_dict2["s3yroot"]
    y_noise2 = loaded_dict2["y_noise"]
    y2 = loaded_dict2["y"]
    std2 = loaded_dict2["std"]


fig, axes = plt.subplots(3,1, figsize=(8,10))

axes[0].scatter(y,std,label = "voter std.", s=0.5)

axes[0].scatter(y,s2yroot,label = r"VA: $\hat{\sigma}_{\theta_2}$", s=0.5)
axes[0].set_xlabel("age")
axes[0].set_ylim((0,15))
axes[0].set_xlim((0,100))

axes[0].scatter(y,s3yroot2,label = r"DVA: $\hat{\sigma}_{\epsilon,\theta_3}$", s=0.5, color="C3")
axes[0].legend()


axes[1].scatter(y,y_noise2,label = "learned normalized noise", s=0.5, color="C3")
axes[1].legend()
axes[1].set_xlabel("age")
axes[2].scatter(std2, s3yroot2)
axes[2].set_xlabel("std")
axes[2].set_ylabel("s3yroot")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("plot_variance.png")
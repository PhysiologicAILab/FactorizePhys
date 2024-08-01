# https://stackoverflow.com/questions/65878921/how-to-make-a-bubble-graph-using-seaborn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# data
models = ["PhysFormer", "PhysNet", "EfficientPhys (SASN)", "EfficientPhys (FSAM)", "FactorizePhys"]
latency = np.array([26.86, 1.36, 1.31, 5.62, 1.94]).astype(np.float64)
accuracy = np.array([0.58, 0.60, 0.72, 0.72, 0.82]).astype(np.float64)
parameters = np.array([7380871, 768583, 2163081, 140655, 51840]).astype(np.float64)

# # create pandas dataframe
# data_list = pd.DataFrame(
#     {'Latency': latency,
#      'Accuracy': accuracy,
#      'Model': models
#      })

# # change size of data points
# minsize = min(parameters)
# maxsize = max(parameters)

# # scatter plot
# # sns.catplot(x="latency", y="accuracy", kind="swarm", hue="models",
# #             sizes=(int(minsize/40.0), int(maxsize/40.0)), data=data_list)

# sns.catplot(x="Latency", y="Accuracy", kind="swarm", hue="Model",
#             sizes=parameters/7000, data=data_list)

# plt.grid()
# plt.savefig("summary_plot.jpg")


df = pd.DataFrame({'Pamaters': parameters, 'Accuracy': accuracy, 'Model': models, 'Latency': latency*10})

sns.scatterplot(x="Pamaters", y="Accuracy", hue='Model', size="Latency", data=df, sizes=(latency*10).astype(np.int16))

plt.show()

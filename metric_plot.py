import matplotlib.pyplot as plt
import pandas as pd

data = {
    'UE_Scale': [100, 150, 200, 300, 400, 500],
    'Latency': [290, 320, 350, 390, 430, 470],
    'Latency_Error': [40, 50, 60, 80, 100, 120]
}

df = pd.DataFrame(data)
plt.errorbar(df['UE_Scale'], df['Latency'], yerr=df['Latency_Error'], 
             fmt='-o', capsize=5, label='Latency (ms)')
plt.xlabel('Number of UEs')
plt.ylabel('Average Latency (ms)')
plt.title('Latency vs. System Scale')
plt.grid(True)
plt.show()
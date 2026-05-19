import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_results.csv")

plt.figure(figsize=(10,6))
plt.bar(df["Model"], df["F1"])

plt.title("Modellerin F1 Skorlarına Göre Karşılaştırılması")
plt.xlabel("Model")
plt.ylabel("F1 Skoru")

plt.ylim(0.7, 0.77)  # farkı büyüt

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("f1_comparison.png")
plt.show()
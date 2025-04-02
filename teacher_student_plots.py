import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


models = [
    {
        "name": "TinyLlama 2bit (7B teacher)",
        "teacher": 7,
        "student": 1.1, 
        "benchmarks": {
            "PPL": 16.94,
            "arc_easy": 36.91,
            "arc_challenge": 20.14,
            "piqa": 60.28,
            "winogrande": 53.99,
            "hellaswag": 33.00,
            "QA Avg": 40.86
        }
    },
    {
        "name": "TinyLlama 2bit (3B Teacher)",
        "teacher": 3,
        "student": 1.1,
        "benchmarks": {
            "PPL": 17.17,
            "arc_easy": 45.16,
            "arc_challenge": 21.84,
            "piqa": 63.22,
            "winogrande": 51.78,
            "hellaswag": 34.14,
            "QA Avg": 43.23
        }
    },
    {
        "name": "Llama 3.2 3B 2bit (7B Teacher)",
        "teacher": 7,
        "student": 3,
        "benchmarks": {
            "PPL": 914841.62,
            "arc_easy": 25.25,
            "arc_challenge": 20.14,
            "piqa": 53.70,
            "winogrande": 48.70,
            "hellaswag": 25.59,
            "QA Avg": 34.68
        }
    },
    {
        "name": "Llama-2-7b-hf_2bit_int (student and teacher both Llama 7B)",
        "teacher": 7,
        "student": 7,
        "benchmarks": {
            "PPL": 7.87,
            "arc_easy": 67.09,
            "arc_challenge": 33.02,
            "piqa": 74.05,
            "winogrande": 61.64,
            "hellaswag": 48.79,
            "QA Avg": 56.92
        }
    },
    {
        "name": "Llama-3.2-3B_2bit_int (Student and teacher both llama 3B)",
        "teacher": 3,
        "student": 3,
        "benchmarks": {
            "PPL": 16.895,
            "arc_easy": 56.44,
            "arc_challenge": 27.39,
            "piqa": 68.82,
            "winogrande": 54.30,
            "hellaswag": 39.76,
            "QA Avg": 47.57
        }
    },
    {
        "name": "Student and teacher both TinyLlama 1.1B 2bit",
        "teacher": 1.1,
        "student": 1.1,
        "benchmarks": {
            "PPL": 23.76,
            "arc_easy": 35.90,
            "arc_challenge": 22.10,
            "piqa": 60.45,
            "winogrande": 52.88,
            "hellaswag": 32.35,
            "QA Avg": 40.71
        }
    }
]

# Define the set of unique teacher and student sizes for the grid.
teacher_sizes = sorted({m["teacher"] for m in models})
student_sizes = sorted({m["student"] for m in models})

benchmarks = ["PPL", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag", "QA Avg"]
heatmap_data = {}

for bm in benchmarks:
    df = pd.DataFrame(index=teacher_sizes, columns=student_sizes, dtype=float)
    
    # For each cell, if there are multiple models, average the values.
    for t in teacher_sizes:
        for s in student_sizes:
            # Find models matching this teacher/student combination
            val = [m["benchmarks"][bm] for m in models if m["teacher"]==t and m["student"]==s and bm in m["benchmarks"]]
            
            df.loc[t, s] = np.mean(val)
    heatmap_data[bm] = df

# Plot heatmaps for each benchmark.
for bm, df in heatmap_data.items():
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": bm})
    ax.set_xlabel("Student Model Size (B)")
    ax.set_ylabel("Teacher Model Size (B)")
    ax.set_title(f"Heatmap for {bm}")
    plt.show()
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of students / exam attempts
n = 1000

# Generate features
tab_switch_count = np.random.randint(0, 10, n)           # 0–9 times
avg_answer_time = np.random.randint(5, 35, n)            # 5–35 seconds
question_revisit_count = np.random.randint(0, 15, n)    # 0–14 revisits
idle_time = np.random.randint(30, 500, n)               # 30–500 seconds
copy_paste_events = np.random.randint(0, 3, n)          # 0–2 events

# Create cheating label using simple rules
# High tab switches, very fast answers, many revisits, high idle time, copy-paste → likely cheating
cheating_label = []
for ts, avg, qr, idle, cp in zip(tab_switch_count, avg_answer_time, question_revisit_count, idle_time, copy_paste_events):
    if ts > 4 or avg < 15 or qr > 8 or idle > 300 or cp > 0:
        cheating_label.append(1)
    else:
        cheating_label.append(0)

# Create DataFrame
df = pd.DataFrame({
    "tab_switch_count": tab_switch_count,
    "avg_answer_time": avg_answer_time,
    "question_revisit_count": question_revisit_count,
    "idle_time": idle_time,
    "copy_paste_events": copy_paste_events,
    "cheating_label": cheating_label
})

# Save dataset as CSV
df.to_csv("exam_behavior.csv", index=False)
print("Dataset generated successfully! Here are first 5 rows:")
print(df.head())

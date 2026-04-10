import matplotlib.pyplot as plt
import numpy as np

# Set up the data from the Test set (RQ5 vs RQ4)
labels = ['no_schema', 'schema/norange', 'schema/compact']
sparql_vals = [0.14, 0.42, 0.42]
sql_vals = [0.19, 1.68, -0.23]
cypher_vals = [0.61, 0.75, 0.05]

x = np.arange(len(labels))
width = 0.25

# Create the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Define colours approximating the original plot
color_sparql = '#367040' # Dark green
color_sql = '#B3A369'    # Khaki/gold
color_cypher = '#5C7596' # Steel blue

# Plot the grouped bars
rects1 = ax.bar(x - width, sparql_vals, width, label='sparql', color=color_sparql)
rects2 = ax.bar(x, sql_vals, width, label='sql', color=color_sql)
rects3 = ax.bar(x + width, cypher_vals, width, label='cypher', color=color_cypher)

# Add a dashed horizontal line at y=0 to indicate the baseline (Single-Language model performance)
ax.axhline(0, color='grey', linestyle='--', linewidth=1.5)

# Add labels, axis titles, and legend
ax.set_title(r'Gain/Loss from Multi-Query-Language Model ($\Delta$ EX)', fontsize=14)
ax.set_ylabel(r'$\Delta$ EX (%)', fontsize=12)
ax.set_xlabel('Schema', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(title='Language', fontsize=11, title_fontsize=12, loc='upper left')

# Adjust layout so labels are not cut off
plt.tight_layout()

# Save as a vector PDF
plt.savefig('RQ5_single-vs-joint-delta_test.pdf', format='pdf', bbox_inches='tight')
plt.close()
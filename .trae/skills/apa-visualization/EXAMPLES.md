# APA Visualization Skill - Usage Examples

This document provides comprehensive examples of how to use the APA Visualization skill to create publication-ready figures.

## Table of Contents

1. [Basic Skill Invocation](#basic-skill-invocation)
2. [Bar Chart Examples](#bar-chart-examples)
3. [Line Graph Examples](#line-graph-examples)
4. [Scatter Plot Examples](#scatter-plot-examples)
5. [Color Palette Examples](#color-palette-examples)
6. [Advanced Examples](#advanced-examples)

---

## Basic Skill Invocation

### Example 1: Simple Bar Chart Request

**User Request:**
```
Create an APA-formatted bar chart comparing test scores across three groups:
- Control: 75
- Treatment A: 85
- Treatment B: 82
```

**What the Skill Provides:**

The skill will generate complete Python code that:
1. Sets up APA-compliant matplotlib parameters
2. Creates a bar chart with proper formatting
3. Adds figure number and title
4. Includes figure notes
5. Saves the figure in multiple formats (PDF, PNG)

**Expected Output:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Setup APA style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Calibri'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Create data
groups = ['Control', 'Treatment A', 'Treatment B']
scores = [75, 85, 82]
colors = ['#4472C4', '#ED7D31', '#A5A5A5']

# Create figure
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

x = np.arange(len(groups))
bars = ax.bar(x, scores, color=colors, width=0.6, 
              edgecolor='black', linewidth=0.8)

# Customize axes
ax.set_ylabel('Test Score')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add title and note
fig.suptitle('Figure 1. Comparison of Test Scores Across Experimental Groups', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 'Note. N = 30 per group. Error bars not shown.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figure_1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_1.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Result:** A publication-ready bar chart saved as `figure_1.pdf` and `figure_1.png`

---

## Bar Chart Examples

### Example 2: Bar Chart with Error Bars

**User Request:**
```
Create an APA bar chart showing reaction times with error bars:
- Condition 1: Mean = 450ms, SD = 50
- Condition 2: Mean = 520ms, SD = 65
- Condition 3: Mean = 380ms, SD = 45
- Condition 4: Mean = 490ms, SD = 55
```

**Expected Code Structure:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
conditions = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4']
means = [450, 520, 380, 490]
stds = [50, 65, 45, 55]

# APA style setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

x = np.arange(len(conditions))
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

# Create bars
bars = ax.bar(x, means, yerr=stds, color=colors, 
              width=0.6, edgecolor='black', linewidth=0.8,
              capsize=4, error_kw={'linewidth': 1})

# Formatting
ax.set_ylabel('Reaction Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=0)
ax.tick_params(direction='in')

# APA title and note
fig.suptitle('Figure 2. Reaction Times Across Experimental Conditions', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Error bars represent standard deviations. N = 40 per condition.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('reaction_times.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

**Visual Result Description:**
- Four bars with distinct colors
- Error bars with caps on top
- Clean, minimal design
- Sans-serif font throughout
- Figure title above, note below

---

### Example 3: Grouped Bar Chart

**User Request:**
```
Create a grouped bar chart comparing two age groups (Young, Old) across three tasks (Task A, Task B, Task C):
Young: 85, 78, 92
Old: 72, 68, 85
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['Task A', 'Task B', 'Task C']
young_scores = [85, 78, 92]
old_scores = [72, 68, 85]

# Setup
fig, ax = plt.subplots(figsize=(10/2.54, 6/2.54))

x = np.arange(len(tasks))
width = 0.35

# Create grouped bars
bars1 = ax.bar(x - width/2, young_scores, width, 
               label='Young', color='#4472C4', edgecolor='black')
bars2 = ax.bar(x + width/2, old_scores, width, 
               label='Old', color='#ED7D31', edgecolor='black')

# Formatting
ax.set_ylabel('Performance Score')
ax.set_xlabel('Task Type')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.tick_params(direction='in')
ax.legend(frameon=False, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 3. Performance Comparison by Age Group and Task', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Young adults (N=50), Older adults (N=45). Higher scores indicate better performance.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('grouped_bar.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Line Graph Examples

### Example 4: Simple Line Graph

**User Request:**
```
Create an APA line graph showing learning progress over 5 sessions:
Session 1: 45
Session 2: 58
Session 3: 72
Session 4: 81
Session 5: 88
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
sessions = [1, 2, 3, 4, 5]
scores = [45, 58, 72, 81, 88]

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Create line with markers
ax.plot(sessions, scores, marker='o', color='#4472C4', 
        linewidth=1.5, markersize=6, markerfacecolor='#4472C4',
        markeredgecolor='black', markeredgewidth=0.5)

# Formatting
ax.set_xlabel('Session Number')
ax.set_ylabel('Performance Score')
ax.set_xticks(sessions)
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 4. Learning Progress Across Training Sessions', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. N = 60 participants. Scores represent mean accuracy percentage.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('learning_curve.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 5: Multi-Line Graph with Error Bars

**User Request:**
```
Create a line graph comparing three treatment groups over 4 time points:
Time: [0, 1, 2, 3]
Control: [50, 52, 51, 53] (SD: 5, 6, 5, 7)
Treatment A: [48, 65, 78, 85] (SD: 6, 7, 6, 5)
Treatment B: [49, 58, 68, 72] (SD: 5, 6, 7, 6)
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
time_points = [0, 1, 2, 3]

control_mean = [50, 52, 51, 53]
control_sd = [5, 6, 5, 7]

treatment_a_mean = [48, 65, 78, 85]
treatment_a_sd = [6, 7, 6, 5]

treatment_b_mean = [49, 58, 68, 72]
treatment_b_sd = [5, 6, 7, 6]

# Setup
fig, ax = plt.subplots(figsize=(10/2.54, 6/2.54))

# Plot lines with error bars
ax.errorbar(time_points, control_mean, yerr=control_sd,
            marker='o', color='#A5A5A5', linewidth=1.5, 
            markersize=6, label='Control', capsize=3)

ax.errorbar(time_points, treatment_a_mean, yerr=treatment_a_sd,
            marker='s', color='#4472C4', linewidth=1.5, 
            markersize=6, label='Treatment A', capsize=3)

ax.errorbar(time_points, treatment_b_mean, yerr=treatment_b_sd,
            marker='^', color='#ED7D31', linewidth=1.5, 
            markersize=6, label='Treatment B', capsize=3)

# Formatting
ax.set_xlabel('Time Point (months)')
ax.set_ylabel('Symptom Score')
ax.set_xticks(time_points)
ax.tick_params(direction='in')
ax.legend(frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 5. Treatment Effects Over Time', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Error bars represent standard deviations. N = 30 per group.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('treatment_effects.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Scatter Plot Examples

### Example 6: Basic Scatter Plot

**User Request:**
```
Create an APA scatter plot showing the relationship between study hours and test scores:
Study Hours: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Test Scores: [65, 68, 72, 75, 78, 82, 85, 88, 91, 94]
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
test_scores = np.array([65, 68, 72, 75, 78, 82, 85, 88, 91, 94])

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Create scatter plot
ax.scatter(study_hours, test_scores, s=50, alpha=0.6, 
           color='#4472C4', edgecolors='black', linewidth=0.5)

# Formatting
ax.set_xlabel('Study Hours')
ax.set_ylabel('Test Score')
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 6. Relationship Between Study Time and Test Performance', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. N = 10 participants. Each point represents one student.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('scatter_basic.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 7: Scatter Plot with Regression Line

**User Request:**
```
Create a scatter plot with a regression line showing the correlation between age and memory score:
Age: [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
Memory Score: [95, 92, 88, 85, 82, 78, 75, 71, 68, 64, 60, 56]
Include the regression equation
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Data
age = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
memory_score = np.array([95, 92, 88, 85, 82, 78, 75, 71, 68, 64, 60, 56])

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Scatter plot
ax.scatter(age, memory_score, s=50, alpha=0.6, 
           color='#4472C4', edgecolors='black', linewidth=0.5)

# Regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(age, memory_score)
line = slope * age + intercept
ax.plot(age, line, color='#ED7D31', linewidth=1.5, linestyle='--',
        label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.3f}')

# Formatting
ax.set_xlabel('Age (years)')
ax.set_ylabel('Memory Score')
ax.tick_params(direction='in')
ax.legend(frameon=False, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 7. Age-Related Decline in Memory Performance', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Dashed line represents linear regression fit. N = 12 participants.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('scatter_regression.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 8: Grouped Scatter Plot

**User Request:**
```
Create a scatter plot comparing two groups (Male, Female) on height vs weight:
Male Height: [170, 175, 180, 185, 190, 172, 178, 183]
Male Weight: [65, 70, 75, 80, 85, 68, 73, 78]
Female Height: [155, 160, 165, 158, 163, 168, 162, 160]
Female Weight: [50, 55, 60, 52, 57, 62, 56, 54]
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
male_height = [170, 175, 180, 185, 190, 172, 178, 183]
male_weight = [65, 70, 75, 80, 85, 68, 73, 78]

female_height = [155, 160, 165, 158, 163, 168, 162, 160]
female_weight = [50, 55, 60, 52, 57, 62, 56, 54]

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Scatter plots
ax.scatter(male_height, male_weight, s=50, alpha=0.6, 
           color='#4472C4', edgecolors='black', linewidth=0.5, 
           label='Male', marker='o')

ax.scatter(female_height, female_weight, s=50, alpha=0.6, 
           color='#ED7D31', edgecolors='black', linewidth=0.5, 
           label='Female', marker='s')

# Formatting
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Weight (kg)')
ax.tick_params(direction='in')
ax.legend(frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 8. Height-Weight Relationship by Gender', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. N = 8 males, N = 8 females. Each point represents one participant.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('scatter_grouped.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Color Palette Examples

### Example 9: Using Different Color Palettes

**User Request:**
```
Create the same bar chart using three different color palettes:
- APA Default
- Okabe-Ito
- Grayscale

Data: Group A: 85, Group B: 92, Group C: 78, Group D: 88
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
groups = ['Group A', 'Group B', 'Group C', 'Group D']
scores = [85, 92, 78, 88]

# Color palettes
palettes = {
    'APA Default': ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'],
    'Okabe-Ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442'],
    'Grayscale': ['#000000', '#333333', '#666666', '#999999']
}

# Create three figures
fig, axes = plt.subplots(1, 3, figsize=(18/2.54, 6/2.54))

for idx, (palette_name, colors) in enumerate(palettes.items()):
    ax = axes[idx]
    x = np.arange(len(groups))
    
    ax.bar(x, scores, color=colors, width=0.6, 
           edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.tick_params(direction='in')
    ax.set_title(palette_name, fontweight='bold', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Figure 9. Same Data with Different Color Palettes', 
             fontsize=12, fontweight='bold', y=1.05)
fig.text(0.5, -0.05, 
         'Note. Left: APA Default palette. Middle: Okabe-Ito palette (colorblind-friendly). Right: Grayscale palette.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('color_palettes_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 10: Previewing Color Palettes

**User Request:**
```
Show me a preview of the Viridis color palette with sample visualizations
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

def preview_viridis_palette():
    """Preview Viridis palette with sample visualizations."""
    
    colors = ['#440154', '#482878', '#3E4A89', '#31688E', 
              '#26828E', '#35B779', '#6DCD59', '#FDE725']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Viridis Palette Preview', fontsize=14, fontweight='bold')
    
    # 1. Color swatches
    ax1 = axes[0, 0]
    for i, color in enumerate(colors):
        ax1.bar(i, 1, color=color, edgecolor='black', linewidth=0.5)
        ax1.text(i, 1.05, color, ha='center', fontsize=8, rotation=45)
    ax1.set_xlim(-0.5, len(colors) - 0.5)
    ax1.set_ylim(0, 1.3)
    ax1.set_title('Color Swatches', fontweight='bold')
    ax1.axis('off')
    
    # 2. Bar chart example
    ax2 = axes[0, 1]
    categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
    values = [85, 92, 78, 88]
    ax2.bar(range(len(categories)), values, color=colors[:len(categories)], 
            edgecolor='black', linewidth=0.8)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Score')
    ax2.set_title('Bar Chart Example', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Line graph example
    ax3 = axes[1, 0]
    x = np.linspace(0, 10, 50)
    for i in range(4):
        y = np.sin(x) + i * 0.5
        ax3.plot(x, y, color=colors[i], linewidth=2, label=f'Line {i+1}')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    ax3.set_title('Line Graph Example', fontweight='bold')
    ax3.legend(frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Scatter plot example
    ax4 = axes[1, 1]
    for i in range(4):
        x_scatter = np.random.randn(25) + i * 2
        y_scatter = np.random.randn(25) + i * 2
        ax4.scatter(x_scatter, y_scatter, color=colors[i], s=50, 
                   alpha=0.6, edgecolors='black', linewidth=0.5, 
                   label=f'Group {i+1}')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    ax4.set_title('Scatter Plot Example', fontweight='bold')
    ax4.legend(frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('viridis_preview.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Run preview
preview_viridis_palette()
```

---

## Advanced Examples

### Example 11: Histogram

**User Request:**
```
Create an APA histogram showing the distribution of reaction times:
Generate 1000 random reaction times with mean=500ms and SD=100ms
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
np.random.seed(42)
reaction_times = np.random.normal(500, 100, 1000)

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Create histogram
n, bins, patches = ax.hist(reaction_times, bins=30, 
                            color='#4472C4', edgecolor='black', 
                            linewidth=0.5, alpha=0.7)

# Formatting
ax.set_xlabel('Reaction Time (ms)')
ax.set_ylabel('Frequency')
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 10. Distribution of Reaction Times', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. N = 1000 trials. Mean = 500 ms, SD = 100 ms.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('histogram.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 12: Box Plot

**User Request:**
```
Create an APA box plot comparing three experimental conditions:
Condition A: [23, 25, 28, 30, 32, 35, 38, 40, 42, 45]
Condition B: [30, 33, 35, 38, 40, 42, 45, 48, 50, 52]
Condition C: [35, 38, 40, 42, 45, 48, 50, 53, 55, 58]
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
condition_a = [23, 25, 28, 30, 32, 35, 38, 40, 42, 45]
condition_b = [30, 33, 35, 38, 40, 42, 45, 48, 50, 52]
condition_c = [35, 38, 40, 42, 45, 48, 50, 53, 55, 58]

data = [condition_a, condition_b, condition_c]
labels = ['Condition A', 'Condition B', 'Condition C']

# Setup
fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))

# Create box plot
bp = ax.boxplot(data, labels=labels, patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1),
                flierprops=dict(marker='o', markerfacecolor='black', 
                               markersize=5, alpha=0.5))

# Color boxes
colors = ['#4472C4', '#ED7D31', '#A5A5A5']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Formatting
ax.set_ylabel('Performance Score')
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 11. Performance Distribution Across Conditions', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Box shows interquartile range. Line represents median. Whiskers extend to 1.5 IQR.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 13: Violin Plot

**User Request:**
```
Create an APA violin plot showing the distribution of scores for four different teaching methods:
Method A: Generate 50 scores with mean=75, SD=10
Method B: Generate 50 scores with mean=80, SD=12
Method C: Generate 50 scores with mean=85, SD=8
Method D: Generate 50 scores with mean=78, SD=15
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
np.random.seed(42)
method_a = np.random.normal(75, 10, 50)
method_b = np.random.normal(80, 12, 50)
method_c = np.random.normal(85, 8, 50)
method_d = np.random.normal(78, 15, 50)

data = [method_a, method_b, method_c, method_d]
labels = ['Method A', 'Method B', 'Method C', 'Method D']

# Setup
fig, ax = plt.subplots(figsize=(10/2.54, 6/2.54))

# Create violin plot
parts = ax.violinplot(data, positions=[1, 2, 3, 4], 
                      showmeans=True, showmedians=True)

# Color violins
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Customize other elements
parts['cmeans'].set_color('red')
parts['cmedians'].set_color('black')
parts['cbars'].set_color('black')
parts['cmaxes'].set_color('black')
parts['cmins'].set_color('black')

# Formatting
ax.set_ylabel('Test Score')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels)
ax.tick_params(direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# APA elements
fig.suptitle('Figure 12. Score Distribution by Teaching Method', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. N = 50 per method. Red line = mean, Black line = median.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('violin_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 14: Heatmap

**User Request:**
```
Create an APA heatmap showing correlation matrix for 5 variables:
Generate a 5x5 correlation matrix with values between -1 and 1
Use the Viridis color palette
```

**Expected Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate correlation matrix
np.random.seed(42)
n_vars = 5
variables = ['Var 1', 'Var 2', 'Var 3', 'Var 4', 'Var 5']

# Create a symmetric correlation matrix
corr_matrix = np.random.randn(n_vars, n_vars)
corr_matrix = np.dot(corr_matrix, corr_matrix.T)
corr_matrix = corr_matrix / np.diag(corr_matrix)[:, np.newaxis]
corr_matrix = corr_matrix / np.diag(corr_matrix)[np.newaxis, :]
np.fill_diagonal(corr_matrix, 1.0)

# Setup
fig, ax = plt.subplots(figsize=(10/2.54, 8/2.54))

# Create heatmap
im = ax.imshow(corr_matrix, cmap='viridis', aspect='auto', 
               vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation', rotation=270, labelpad=15)

# Add labels
ax.set_xticks(np.arange(n_vars))
ax.set_yticks(np.arange(n_vars))
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)
ax.tick_params(direction='in')

# Add correlation values as text
for i in range(n_vars):
    for j in range(n_vars):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', 
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                       fontsize=8)

# APA elements
fig.suptitle('Figure 13. Correlation Matrix Heatmap', 
             fontsize=12, fontweight='bold', y=1.02)
fig.text(0.5, -0.05, 
         'Note. Correlation values range from -1 to 1. Darker colors indicate stronger correlations.', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('heatmap.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Tips for Best Results

### 1. Be Specific About Data
- Provide exact values when possible
- Specify units (e.g., "Reaction Time (ms)")
- Include sample sizes for notes

### 2. Specify Color Palette
- Choose appropriate palette for your use case
- Consider colorblind accessibility
- Match journal requirements

### 3. Include Context
- Add meaningful figure notes
- Explain abbreviations
- Provide sample sizes

### 4. Request Multiple Formats
- PDF for vector graphics
- PNG for raster graphics
- Specify resolution if needed

### 5. Customize as Needed
- Request specific dimensions
- Ask for error bars when appropriate
- Specify legend placement

---

## Common Customization Requests

### Adjusting Figure Size
```
Create a bar chart with double-column width (17.5 cm)
```

### Adding Statistical Annotations
```
Add significance stars to the bar chart comparing groups
```

### Custom Colors
```
Use my institution's colors: #0066CC, #FF6600, #00CC66
```

### Multiple Panels
```
Create a 2x2 figure with four different plots
```

---

## Summary

The APA Visualization skill provides:
- ✅ Automatic APA formatting
- ✅ Multiple chart types
- ✅ 8 colorblind-friendly palettes
- ✅ High-quality output (300 DPI)
- ✅ Proper figure numbering and notes
- ✅ Sans-serif fonts throughout
- ✅ Clean, minimal design

Simply describe your data and requirements, and the skill will generate publication-ready figures that comply with APA guidelines.

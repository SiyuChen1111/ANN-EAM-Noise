---
name: "apa-visualization"
description: "Creates APA-compliant visualizations with proper formatting. Invoke when user needs to create publication-ready figures, charts, or plots following APA style guidelines."
---

# APA Visualization Formatter

This skill helps create publication-ready visualizations that comply with American Psychological Association (APA) style guidelines for academic papers and journals.

## When to Invoke This Skill

Use this skill when:
- Creating figures for academic papers or journal submissions
- User requests APA-compliant charts or plots
- Formatting visualizations for publication
- Converting existing plots to APA style
- User asks about APA figure formatting requirements

## APA Visualization Requirements

### 1. Figure Components

Every APA figure should include:

#### Figure Number and Title
- **Location**: Above the figure
- **Format**: Bold "Figure X" followed by title on same line
- **Example**: **Figure 1**. Comparison of Model Performance Across Different Conditions
- **Title**: Italicized, brief but descriptive
- **Capitalization**: Sentence case for title

#### Figure Notes
- **Location**: Below the figure
- **Format**: Italicized "Note." followed by description
- **Content**: Explain abbreviations, provide context, cite source if adapted
- **Example**: *Note.* Error bars represent standard errors. N = 150 for each condition.

### 2. Font Specifications

- **Font Family**: Sans-serif fonts (Arial, Helvetica, Calibri)
- **Font Size**:
  - Figure title: 12-14 pt
  - Axis labels: 11-12 pt
  - Tick labels: 10-11 pt
  - Legend text: 10-11 pt
  - Figure notes: 10 pt
- **Consistency**: Use same font throughout the manuscript

### 3. Axis Formatting

#### Axis Labels
- **Position**: Parallel to axis
- **Format**: Clear, concise descriptions
- **Units**: Include in parentheses when applicable
- **Example**: "Reaction Time (ms)" or "Accuracy (%)"

#### Axis Ticks
- **Tick marks**: Point inward
- **Tick labels**: Clear, readable size
- **Scale**: Linear unless logarithmic scale is justified
- **Range**: Minimize white space while showing all data clearly

#### Axis Lines
- **Style**: Solid black lines
- **Width**: 0.5-1.0 pt
- **Box**: Optional, but if used, maintain consistent line width

### 4. Color and Style Guidelines

#### Color Usage
- **Primary approach**: Use grayscale when possible
- **Color alternatives**: Use patterns, textures, or different line styles
- **Color palette** (if color is necessary):
  - Use colorblind-friendly palettes
  - Recommended: viridis, ColorBrewer qualitative palettes
  - Avoid: red-green combinations (colorblind accessibility)
- **Contrast**: Ensure sufficient contrast for printing

#### Line Styles
- **Line width**: 1.0-2.0 pt for main data
- **Styles**: Solid, dashed, dotted, dash-dot
- **Differentiation**: Use different markers (circles, squares, triangles)

### 4.5. Color Palette Options and Preview

This skill provides multiple colorblind-friendly color palettes for APA-compliant visualizations. Each palette is designed to meet accessibility standards while maintaining professional aesthetics.

#### Available Color Palettes

**1. APA Default Palette**
- **Colors**: `['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']`
- **Use Case**: General purpose, suitable for most academic journals
- **Colorblind Safe**: Yes (deuteranopia/protanopia friendly)
- **Preview**: Blue, Orange, Gray, Gold, Light Blue, Green

**2. Viridis Palette**
- **Colors**: `['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#35B779', '#6DCD59', '#FDE725']`
- **Use Case**: Sequential data, heatmaps, continuous variables
- **Colorblind Safe**: Yes (all types)
- **Preview**: Dark Purple → Purple → Blue → Teal → Green → Yellow

**3. ColorBrewer Qualitative (Set2)**
- **Colors**: `['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']`
- **Use Case**: Categorical data, multiple groups
- **Colorblind Safe**: Yes
- **Preview**: Teal, Orange, Blue, Pink, Green, Yellow, Tan, Gray

**4. Okabe-Ito Palette**
- **Colors**: `['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']`
- **Use Case**: Universal design, maximum accessibility
- **Colorblind Safe**: Yes (all types)
- **Preview**: Orange, Sky Blue, Green, Yellow, Blue, Vermillion, Red-Purple, Gray

**5. Grayscale Palette**
- **Colors**: `['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#E5E5E5']`
- **Use Case**: Print publications, journals requiring grayscale
- **Colorblind Safe**: Yes (all types)
- **Preview**: Black → Dark Gray → Medium Gray → Light Gray → Very Light Gray

**6. Nature Publishing Group Palette**
- **Colors**: `['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']`
- **Use Case**: Nature journals, high-impact publications
- **Colorblind Safe**: Partially (avoid red-green combinations)
- **Preview**: Red, Cyan, Green, Dark Blue, Salmon, Purple-Gray

**7. Science Magazine Palette**
- **Colors**: `['#228B22', '#FF6347', '#4169E1', '#FFD700', '#8B4513', '#9370DB']`
- **Use Case**: Science journal submissions
- **Colorblind Safe**: Partially
- **Preview**: Forest Green, Tomato, Royal Blue, Gold, Saddle Brown, Medium Purple

**8. Wong Palette (Colorblind-Friendly)**
- **Colors**: `['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']`
- **Use Case**: Scientific publications, presentations
- **Colorblind Safe**: Yes (all types)
- **Preview**: Black, Orange, Sky Blue, Green, Yellow, Blue, Vermillion, Red-Purple

#### Color Palette Preview Function

```python
import matplotlib.pyplot as plt
import numpy as np

def preview_color_palette(palette_name='apa_default'):
    """
    Preview a color palette with sample visualizations.
    
    Parameters:
    -----------
    palette_name : str, name of the palette to preview
        Options: 'apa_default', 'viridis', 'colorbrewer_set2', 'okabe_ito',
                'grayscale', 'nature', 'science', 'wong'
    """
    palettes = {
        'apa_default': ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'],
        'viridis': ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#35B779', '#6DCD59', '#FDE725'],
        'colorbrewer_set2': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3'],
        'okabe_ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'],
        'grayscale': ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#E5E5E5'],
        'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4'],
        'science': ['#228B22', '#FF6347', '#4169E1', '#FFD700', '#8B4513', '#9370DB'],
        'wong': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    }
    
    colors = palettes.get(palette_name, palettes['apa_default'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{palette_name.replace("_", " ").title()} Palette Preview', 
                 fontsize=14, fontweight='bold')
    
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
    categories = ['Group A', 'Group B', 'Group C', 'Group D']
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
    for i in range(min(4, len(colors))):
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
    for i in range(min(4, len(colors))):
        x_scatter = np.random.randn(25) + i * 2
        y_scatter = np.random.randn(25) + i * 2
        ax4.scatter(x_scatter, y_scatter, color=colors[i], s=50, 
                   alpha=0.6, edgecolors='black', linewidth=0.5, label=f'Group {i+1}')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    ax4.set_title('Scatter Plot Example', fontweight='bold')
    ax4.legend(frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return colors

# Preview all palettes
def preview_all_palettes():
    """Preview all available color palettes side by side."""
    palette_names = ['apa_default', 'viridis', 'colorbrewer_set2', 'okabe_ito',
                    'grayscale', 'nature', 'science', 'wong']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('All Available Color Palettes', fontsize=16, fontweight='bold')
    
    for idx, palette_name in enumerate(palette_names):
        ax = axes[idx // 4, idx % 4]
        
        palettes = {
            'apa_default': ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'],
            'viridis': ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#35B779', '#6DCD59', '#FDE725'],
            'colorbrewer_set2': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3'],
            'okabe_ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'],
            'grayscale': ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#E5E5E5'],
            'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4'],
            'science': ['#228B22', '#FF6347', '#4169E1', '#FFD700', '#8B4513', '#9370DB'],
            'wong': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        }
        
        colors = palettes[palette_name]
        
        for i, color in enumerate(colors):
            ax.bar(i, 1, color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlim(-0.5, len(colors) - 0.5)
        ax.set_ylim(0, 1.2)
        ax.set_title(palette_name.replace('_', ' ').title(), fontweight='bold', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# preview_color_palette('okabe_ito')
# preview_all_palettes()
```

#### How to Specify Color Palette

When creating visualizations, you can specify the color palette in several ways:

**Method 1: Use Predefined Palette Name**
```python
# In your request, specify the palette name:
"Create a bar chart using the Okabe-Ito color palette"
"Use the Nature Publishing Group palette for this line graph"
```

**Method 2: Provide Custom Colors**
```python
# Provide your own color list:
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
```

**Method 3: Request Palette Recommendation**
```python
# Ask for recommendation based on your use case:
"What color palette would you recommend for a bar chart with 6 categories?"
"Which palette is best for a journal submission to Nature?"
```

#### Color Palette Selection Guide

| Use Case | Recommended Palette | Reason |
|----------|-------------------|---------|
| General academic papers | APA Default | Professional, widely accepted |
| Colorblind accessibility | Okabe-Ito or Wong | Maximum accessibility for all types |
| Sequential/continuous data | Viridis | Perceptually uniform |
| Categorical data (many groups) | ColorBrewer Set2 | Distinct, harmonious colors |
| Print publications | Grayscale | No color printing required |
| Nature journals | Nature Publishing Group | Matches journal style |
| Science journal | Science Magazine | Matches journal style |
| Presentations | Wong or Okabe-Ito | High contrast, visible from distance |

#### Accessibility Testing

All color palettes have been tested for:
- **Deuteranopia** (red-green colorblindness)
- **Protanopia** (red weakness)
- **Tritanopia** (blue-yellow colorblindness)
- **Achromatopsia** (total colorblindness)

Palettes marked as "Colorblind Safe: Yes (all types)" are safe for all forms of color vision deficiency.

### 5. Chart Elements

#### Bar Charts
- **Bar width**: Consistent, not too wide or narrow
- **Spacing**: Small gap between bars in same group
- **Error bars**: Include when showing variability
- **Error bar style**: Caps on both ends, clear but not overwhelming

#### Line Graphs
- **Data points**: Clearly marked with distinct markers
- **Line connections**: Straight lines between points (not smoothed)
- **Multiple lines**: Different styles/colors with clear legend

#### Scatter Plots
- **Marker size**: Visible but not overlapping excessively
- **Transparency**: Use alpha for overlapping points
- **Trend lines**: Include regression line if relevant

#### Histograms
- **Bin width**: Appropriate for data distribution
- **Bars**: Adjacent (no gaps)
- **Y-axis**: Frequency or density

### 6. Legend

- **Position**: Within plot area (preferred) or to the right
- **Border**: Optional thin border
- **Background**: White or transparent
- **Order**: Logical order matching data presentation
- **Labels**: Concise, avoid redundancy with axis labels

### 7. Image Quality

- **Resolution**: Minimum 300 DPI for print, 150 DPI for web
- **Format**: 
  - Vector formats preferred: PDF, SVG, EPS
  - Raster formats: PNG (high-res), TIFF
  - Avoid: JPEG (compression artifacts)
- **Size**: 
  - Single column: 8.5 cm (3.35 in) wide
  - Double column: 17.5 cm (6.89 in) wide
  - Maximum height: 23 cm (9 in)

### 8. Accessibility

- **Colorblind-friendly**: Use palettes accessible to colorblind readers
- **Patterns**: Combine color with patterns for differentiation
- **Labels**: Direct labeling preferred over color-only coding
- **Contrast**: Ensure high contrast between elements

## Python Implementation Examples

### Matplotlib APA Style Setup

```python
import matplotlib.pyplot as plt
import numpy as np

def setup_apa_style():
    """Configure matplotlib for APA-compliant figures."""
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
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

def save_apa_figure(fig, filename, fig_num, title, note=None):
    """
    Save figure with APA formatting.
    
    Parameters:
    -----------
    fig : matplotlib figure object
    filename : str, output filename (without extension)
    fig_num : int, figure number
    title : str, figure title
    note : str, optional figure note
    """
    # Add figure number and title
    fig.suptitle(f'Figure {fig_num}. {title}', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    # Add note if provided
    if note:
        fig.text(0.5, -0.05, f'Note. {note}', 
                ha='center', fontsize=10, style='italic')
    
    # Save in multiple formats
    for ext in ['.pdf', '.png']:
        fig.savefig(filename + ext, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
```

### Bar Chart Example

```python
def create_apa_bar_chart(data, labels, ylabel, fig_num, title, 
                         note=None, error_bars=None, colors=None):
    """
    Create APA-compliant bar chart.
    
    Parameters:
    -----------
    data : array-like, bar heights
    labels : list, x-axis labels
    ylabel : str, y-axis label
    fig_num : int, figure number
    title : str, figure title
    note : str, optional figure note
    error_bars : array-like, optional error bar heights
    colors : list, optional bar colors
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))
    
    x = np.arange(len(labels))
    
    if colors is None:
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    bars = ax.bar(x, data, color=colors[:len(labels)], 
                  width=0.6, linewidth=0.8, edgecolor='black')
    
    if error_bars is not None:
        ax.errorbar(x, data, yerr=error_bars, 
                   fmt='none', ecolor='black', capsize=4, capthick=1)
    
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(direction='in')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_apa_figure(fig, f'figure_{fig_num}', fig_num, title, note)
    
    return fig, ax
```

### Line Graph Example

```python
def create_apa_line_graph(x_data, y_data_list, labels, xlabel, ylabel,
                          fig_num, title, note=None, markers=None):
    """
    Create APA-compliant line graph.
    
    Parameters:
    -----------
    x_data : array-like, x-axis values
    y_data_list : list of arrays, y-axis values for each line
    labels : list, legend labels for each line
    xlabel : str, x-axis label
    ylabel : str, y-axis label
    fig_num : int, figure number
    title : str, figure title
    note : str, optional figure note
    markers : list, optional marker styles
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))
    
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']
    
    for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
        ax.plot(x_data, y_data, 
               marker=markers[i % len(markers)],
               color=colors[i % len(colors)],
               label=label,
               linewidth=1.5,
               markersize=6)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction='in')
    ax.legend(frameon=False, loc='best')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_apa_figure(fig, f'figure_{fig_num}', fig_num, title, note)
    
    return fig, ax
```

### Scatter Plot Example

```python
def create_apa_scatter_plot(x_data, y_data, xlabel, ylabel, 
                            fig_num, title, note=None, 
                            regression_line=False):
    """
    Create APA-compliant scatter plot.
    
    Parameters:
    -----------
    x_data : array-like, x-axis values
    y_data : array-like, y-axis values
    xlabel : str, x-axis label
    ylabel : str, y-axis label
    fig_num : int, figure number
    title : str, figure title
    note : str, optional figure note
    regression_line : bool, whether to add regression line
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8.5/2.54, 6/2.54))
    
    ax.scatter(x_data, y_data, alpha=0.6, s=50, 
              color='#4472C4', edgecolors='black', linewidth=0.5)
    
    if regression_line:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_data), max(x_data), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, 
               label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        ax.legend(frameon=False)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction='in')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_apa_figure(fig, f'figure_{fig_num}', fig_num, title, note)
    
    return fig, ax
```

## Common Mistakes to Avoid

1. **Title Placement**: Don't place title below the figure
2. **Font Consistency**: Don't mix different font families
3. **3D Effects**: Avoid 3D charts unless absolutely necessary
4. **Excessive Decoration**: Minimize chartjunk (unnecessary visual elements)
5. **Missing Labels**: Always label axes clearly
6. **Poor Resolution**: Don't use low-resolution images
7. **Inconsistent Scales**: Use consistent scales across similar figures
8. **Color Overuse**: Prefer grayscale or minimal color palette
9. **Missing Error Bars**: Include error bars for statistical data
10. **Cluttered Legends**: Simplify or use direct labeling

## Checklist for APA Compliance

Before finalizing any figure, verify:

- [ ] Figure number and title above figure (bold number, italicized title)
- [ ] Figure note below figure (italicized "Note.")
- [ ] Sans-serif font throughout
- [ ] Appropriate font sizes (10-14 pt)
- [ ] Clear axis labels with units
- [ ] Readable tick labels
- [ ] Top and right spines removed (optional but recommended)
- [ ] Legend positioned appropriately
- [ ] Colorblind-friendly palette (if using color)
- [ ] Minimum 300 DPI resolution
- [ ] Appropriate file format (PDF/PNG/TIFF)
- [ ] Correct dimensions for journal requirements
- [ ] Error bars included where appropriate
- [ ] All abbreviations explained in note
- [ ] Consistent style with other figures in manuscript

## Integration with Academic Workflow

### For Journal Submissions
1. Check target journal's specific figure requirements
2. Adjust dimensions to match journal specifications
3. Verify file format requirements (TIFF, EPS, etc.)
4. Ensure figure references in text are correct

### For Presentations
1. Increase font sizes for readability (14-18 pt)
2. Use higher contrast colors
3. Simplify complex figures
4. Consider widescreen format (16:9)

### For Reproducibility
1. Save figure generation code
2. Document all parameters
3. Use random seeds for reproducible examples
4. Version control figure files

## Additional Resources

- APA Publication Manual (7th Edition) - Chapter 7: Tables and Figures
- Journal-specific author guidelines
- ColorBrewer: https://colorbrewer2.org/ (colorblind-safe palettes)
- Matplotlib style sheets for academic publishing

## Notes

- Always check specific journal requirements, as they may override APA guidelines
- Some journals require figures as separate files, others embedded in manuscript
- Vector formats (PDF, SVG) are preferred for line graphs and charts
- Raster formats (PNG, TIFF) may be required for complex images or photographs
- When in doubt, consult the journal's author guidelines or contact the editorial office

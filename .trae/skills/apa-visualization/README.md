# APA Visualization Skill

## Overview

`apa-visualization` is a specialized skill for creating visualizations that comply with the American Psychological Association (APA) style guidelines. This skill helps users generate publication-ready figures that fully meet the requirements of the APA Publication Manual (7th Edition).

## Table of Contents

- [Purpose](#purpose)
- [How to Invoke](#how-to-invoke)
- [Core Features](#core-features)
- [Color Palette Options](#color-palette-options)
- [Supported Visualization Types](#supported-visualization-types)
- [Usage Examples](#usage-examples)
- [APA Format Checklist](#apa-format-checklist)
- [Technical Implementation](#technical-implementation)
- [Documentation](#documentation)
- [Important Notes](#important-notes)
- [Related Resources](#related-resources)

## Purpose

### Main Features

- **Create APA-compliant figures**: Generate bar charts, line graphs, scatter plots, and more that meet academic journal publication standards
- **Format existing visualizations**: Convert existing plots to APA style
- **Provide APA guideline guidance**: Offer detailed APA figure formatting requirements and best practices
- **Color palette management**: 8 carefully curated colorblind-friendly palettes
- **Preview functionality**: Preview color palettes before using them

### Use Cases

- Writing academic papers or journal submissions
- Creating APA-compliant charts or plots
- Formatting visualizations for publication
- Converting existing plots to APA style
- Learning about APA figure formatting requirements
- Creating accessible visualizations for colorblind readers

## How to Invoke

### Method 1: Direct Request for APA-Formatted Figures

Simply state your need for APA-formatted figures in your conversation, and this skill will be automatically triggered.

**Example Requests:**
```
Please create an APA-formatted bar chart showing performance comparison across different experimental groups
```

```
I need an APA-style line graph to display time series data
```

### Method 2: Explicitly Invoke the Skill

Mention APA visualization or APA format requirements explicitly in your request.

**Example Requests:**
```
Use the apa-visualization skill to help me create a scatter plot
```

```
Help me convert this chart to APA format
```

### Method 3: Ask About APA Format Requirements

Ask questions related to APA figure formatting.

**Example Requests:**
```
What are the requirements for APA-formatted figures?
```

```
How should I format figure titles and notes in APA style?
```

## Core Features

### 1. Complete APA Format Support

- **Figure Number and Title**: Automatically adds APA-compliant figure numbering and title formatting
- **Figure Notes**: Provides standardized figure note formatting
- **Font Specifications**: Uses sans-serif fonts (Arial, Helvetica, Calibri)
- **Font Size Settings**: Title 12-14pt, axis labels 11-12pt, tick labels 10-11pt

### 2. Multiple Chart Types

- **Bar Charts**: Supports error bars, grouped bar charts
- **Line Graphs**: Supports multiple lines, different marker styles
- **Scatter Plots**: Supports regression lines, transparency settings
- **Histograms**: Supports frequency and density display
- **Box Plots**: Distribution visualization
- **Violin Plots**: Density distribution
- **Heatmaps**: Correlation matrices and 2D data

### 3. Professional Output Quality

- **High Resolution**: Default 300 DPI, meeting print requirements
- **Multiple Formats**: Supports PDF, PNG, SVG, TIFF, and other formats
- **Journal Dimensions**: Complies with single-column (8.5cm) and double-column (17.5cm) width requirements

### 4. Accessibility Design

- **Colorblind-Friendly**: Uses colorblind-friendly color palettes
- **Pattern Differentiation**: Combines color with patterns for data differentiation
- **High Contrast**: Ensures sufficient contrast between elements

## Color Palette Options

This skill provides 8 carefully curated color palettes designed for academic publications and accessibility. All palettes have been tested for colorblind accessibility.

### Available Palettes

1. **APA Default Palette**
   - General purpose, suitable for most academic journals
   - Colorblind safe for deuteranopia/protanopia
   - Colors: Blue, Orange, Gray, Gold, Light Blue, Green

2. **Viridis Palette**
   - Best for sequential data, heatmaps, continuous variables
   - Colorblind safe for all types
   - Perceptually uniform progression

3. **ColorBrewer Set2**
   - Ideal for categorical data with multiple groups
   - Colorblind safe
   - Distinct, harmonious colors

4. **Okabe-Ito Palette**
   - Universal design with maximum accessibility
   - Colorblind safe for all types
   - Recommended for presentations

5. **Grayscale Palette**
   - Perfect for print publications
   - Colorblind safe for all types
   - No color printing required

6. **Nature Publishing Group Palette**
   - Matches Nature journal style
   - Suitable for high-impact publications

7. **Science Magazine Palette**
   - Matches Science journal style
   - Designed for Science submissions

8. **Wong Palette**
   - Scientific publications and presentations
   - Colorblind safe for all types
   - High contrast for visibility

### How to Specify Color Palette

**Method 1: Specify by Name**
```
Create a bar chart using the Okabe-Ito color palette
```

**Method 2: Request Recommendation**
```
What color palette would you recommend for a bar chart with 6 categories?
```

**Method 3: Provide Custom Colors**
```
Use these colors: #FF6B6B, #4ECDC4, #45B7D1, #FFA07A
```

### Color Palette Selection Guide

| Use Case | Recommended Palette |
|----------|-------------------|
| General academic papers | APA Default |
| Maximum colorblind accessibility | Okabe-Ito or Wong |
| Sequential/continuous data | Viridis |
| Categorical data (many groups) | ColorBrewer Set2 |
| Print publications | Grayscale |
| Nature journal submission | Nature Publishing Group |
| Science journal submission | Science Magazine |
| Presentations | Wong or Okabe-Ito |

### Preview Palettes

You can preview any color palette before using it:
```
Show me a preview of the Viridis color palette
Preview all available color palettes
```

## Supported Visualization Types

### Basic Charts

| Chart Type | Features | Best For |
|------------|----------|----------|
| **Bar Chart** | Error bars, grouping, custom colors | Comparing categories |
| **Line Graph** | Multiple lines, markers, error bars | Time series, trends |
| **Scatter Plot** | Regression lines, grouping, transparency | Correlations, distributions |
| **Histogram** | Custom bins, density/frequency | Data distribution |

### Advanced Visualizations

| Chart Type | Features | Best For |
|------------|----------|----------|
| **Box Plot** | Multiple groups, outliers | Distribution comparison |
| **Violin Plot** | Density curves, quartiles | Distribution shape |
| **Heatmap** | Color mapping, annotations | Correlation matrices |
| **Grouped Charts** | Multi-panel figures | Complex comparisons |

## Usage Examples

### Example 1: Create APA-Formatted Bar Chart

**User Request:**
```
Please create an APA-formatted bar chart comparing accuracy across three experimental groups with the following data:
- Control Group: 75%
- Experimental Group 1: 85%
- Experimental Group 2: 82%
```

**Skill Will Provide:**
- Complete Python code implementation
- APA-compliant figure title and notes
- Correct font, color, and size settings
- High-quality output files

### Example 2: Create Line Graph with Error Bars

**User Request:**
```
Create an APA-style line graph showing measurements across four time points, including error bars
```

**Skill Will Provide:**
- Line graph code with error bars
- Clear legend and axis labels
- APA-formatted figure notes
- Correct representation of standard error bars

### Example 3: Convert Existing Chart

**User Request:**
```
I have a matplotlib chart, please help me convert it to APA format
```

**Skill Will Provide:**
- Chart formatting adjustment suggestions
- Complete APA formatting code
- Correct format for titles, notes, and labels
- Best practices for output files

### Example 4: Use Specific Color Palette

**User Request:**
```
Create a bar chart using the Okabe-Ito color palette for colorblind accessibility
```

**Skill Will Provide:**
- Bar chart with Okabe-Ito palette
- Colorblind-friendly visualization
- Preview of color palette if requested

## APA Format Checklist

When using this skill to create figures, the following APA requirements are automatically ensured:

- ✅ Figure number and title positioned above the figure
- ✅ Figure note positioned below the figure
- ✅ Sans-serif font used throughout
- ✅ Appropriate font size settings
- ✅ Clear axis labels with units
- ✅ Readable tick labels
- ✅ Top and right spines removed
- ✅ Legend positioned appropriately
- ✅ Colorblind-friendly color palette
- ✅ Minimum 300 DPI resolution
- ✅ Appropriate file format
- ✅ Dimensions meeting journal requirements
- ✅ Error bars included (where applicable)
- ✅ All abbreviations explained in notes

## Technical Implementation

This skill is implemented based on Python's matplotlib library, providing:

### Core Functions

- `setup_apa_style()`: Configure matplotlib APA style parameters
- `save_apa_figure()`: Save APA-formatted figures
- `create_apa_bar_chart()`: Create APA-formatted bar charts
- `create_apa_line_graph()`: Create APA-formatted line graphs
- `create_apa_scatter_plot()`: Create APA-formatted scatter plots

### Color Palette Functions

- `preview_color_palette()`: Preview a specific color palette
- `preview_all_palettes()`: Preview all available palettes

### Supported Libraries

- matplotlib >= 3.0
- numpy >= 1.15
- scipy (optional, for regression lines)

## Documentation

### Repository Structure

```
APA-visulazation--skills/
├── SKILL.md           # Complete skill definition and guidelines
├── README.md          # This file - quick start guide
├── EXAMPLES.md        # Comprehensive usage examples with code
└── (output files)     # Generated figures (PDF, PNG, etc.)
```

### Documentation Files

- **SKILL.md**: Detailed APA formatting requirements, color palette specifications, and implementation details
- **EXAMPLES.md**: 14 comprehensive examples covering all visualization types with complete code
- **README.md**: Quick start guide and overview (this file)

### Getting Started

1. **Basic Request**: Simply describe your visualization needs
   ```
   Create an APA bar chart showing [your data]
   ```

2. **Specify Requirements**: Add details as needed
   ```
   Create an APA bar chart with error bars using the Okabe-Ito palette
   ```

3. **Review Output**: The skill provides complete, runnable code

4. **Customize**: Request adjustments as needed
   ```
   Make the figure wider and add a legend
   ```

## Important Notes

1. **Journal-Specific Requirements**: Some journals may have specific figure requirements; prioritize journal guidelines
2. **File Formats**: Vector formats (PDF, SVG) are suitable for line graphs and charts; raster formats (PNG, TIFF) are suitable for complex images
3. **Color Usage**: Prefer grayscale; use colorblind-friendly color palettes when necessary
4. **Consistency**: Ensure all figures in the manuscript have consistent style
5. **Resolution**: Always use minimum 300 DPI for print publications
6. **Accessibility**: Consider colorblind readers when choosing palettes

## Related Resources

### Official Guidelines

- APA Publication Manual (7th Edition) - Chapter 7: Tables and Figures
- Journal-specific author guidelines

### Color Resources

- ColorBrewer: https://colorbrewer2.org/ (colorblind-safe palettes)
- Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html

### Technical Resources

- Matplotlib documentation: https://matplotlib.org/
- Matplotlib style sheets for academic publishing
- Python visualization best practices

## Quick Reference

### Common Requests

| Task | Example Request |
|------|----------------|
| Create bar chart | "Create an APA bar chart comparing groups A, B, C" |
| Add error bars | "Add error bars to the bar chart" |
| Change colors | "Use the Viridis color palette" |
| Adjust size | "Make it double-column width" |
| Add regression | "Add a regression line to the scatter plot" |
| Preview palette | "Show me the Okabe-Ito palette preview" |

### Output Formats

- **PDF**: Vector format, best for line graphs and charts
- **PNG**: Raster format, high resolution (300 DPI)
- **SVG**: Vector format, editable in vector graphics software
- **TIFF**: Raster format, required by some journals

## Skill Information

- **Skill Name**: apa-visualization
- **Version**: 1.1
- **Dependencies**: Python 3.x, matplotlib, numpy
- **Author**: APA Visualization Team
- **Last Updated**: 2024

## Support

For detailed examples and code snippets, see [EXAMPLES.md](EXAMPLES.md)

For complete APA formatting guidelines and specifications, see [SKILL.md](SKILL.md)

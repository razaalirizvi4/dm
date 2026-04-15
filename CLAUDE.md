# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Typical Development Commands

This repository focuses on data science and machine learning experiments for IoT network intrusion detection. Common development workflows include:

- **Run classification pipeline**: `python classification.py` - Trains and evaluates multiple classifiers on the IoT dataset
- **Run data cleaning comparisons**: Execute cells in `DataCleaning_Comparisons.ipynb` notebook to compare preprocessing techniques
- **Run preprocessing comparisons**: Use the preprocessing-comparator agent to systematically evaluate different preprocessing pipelines
- **Run simple data inspection**: `python main.py` - Loads and displays basic dataset information
- **Run utility scripts**: Individual Python scripts like `check_cols.py`, `check_data.py`, `fix_lda.py` for specific data examination tasks

## High‑Level Project Structure

- `AI_Powered_IoT_Network_Intrusion_Detection_Dataset.csv` – Primary dataset containing IoT network traffic features and intrusion labels
- `classification.py` – Main classification pipeline implementing 6 different algorithms (Decision Tree, Naive Bayes, KNN, SVM, Random Forest, MLP) with SMOTE for class imbalance
- `DataCleaning_Comparisons.ipynb` – Jupyter notebook demonstrating 9 data-cleaning techniques (imputation, outlier removal, binning, etc.)
- `Preprocessing_Evaluation.ipynb` – Additional notebook for preprocessing evaluation experiments
- `.claude/agents/preprocessing-comparator.md` – Specialized agent for systematic comparison of data preprocessing techniques
- `.claude/agent-memory/` – Persistent storage for agent knowledge and learned patterns
- `results/` directory – Contains generated outputs including:
  - Performance visualizations (metrics_comparison.png, roc_curves.png, confusion_matrices.png, etc.)
  - Numerical results (results.json, summary_table.csv)
  - Feature importance and training time comparisons
- Utility Python scripts: `main.py`, `check_cols.py`, `check_data.py`, `fix_lda.py`, `_inspect.py` for various data inspection tasks

### Architecture Overview

This repository follows a **data-centric experimental architecture** where:

1. **Core Dataset**: The CSV file serves as the immutable source of truth; all processing creates derived copies rather than modifying the original
2. **Experimental Scripts**: Python scripts and Jupyter notebooks implement specific analyses or comparisons
3. **Agent-Based Tools**: Specialized Claude agents (like preprocessing-comparator) provide reusable frameworks for common data science tasks
4. **Results Organization**: All outputs are systematically saved to the `results/` directory with standardized naming
5. **Memory System**: Agents can accumulate and reuse knowledge across sessions through the `.claude/agent-memory/` directory

### Key Development Patterns

- **Data Preservation**: Original dataset is never modified; techniques create suffixed copies (e.g., `*_mean_imputed.csv`)
- **Experiment Tracking**: Results are automatically saved with timestamps and organized by experiment type
- **Reusable Components**: Common patterns like data loading, preprocessing, and evaluation are encapsulated for reuse
- **Visualization Focus**: Strong emphasis on generating comparative visualizations to aid interpretation
- **Agent Collaboration**: The preprocessing-comparator agent can be invoked to perform systematic evaluations that would be tedious to do manually

## Adding Future Documentation

When adding new components to this repository:

- **New Python scripts**: Place in root directory and document their specific purpose in this file
- **New notebooks**: Add to root directory and describe their experimental focus
- **New agents**: Create under `.claude/agents/` following the preprocessing-comparator template
- **Updated structure**: If introducing `src/` or `notebooks/` directories, update the directory layout description accordingly
- **Command documentation**: Add any new common commands to the Typical Development Commands section

---
*Updated to reflect the actual data science and machine learning focus of this repository*
# siRNA-Effectiveness-ML
This project performs binary classification on a dataset (data.csv) using Logistic Regression and Random Forest classifiers.
## Requirements
Install the required Python packages using pip:

```
pip install pandas scikit-learn matplotlib seaborn graphviz
```

**Note:** On macOS, graphviz may need to be installed separately (e.g. using homebrew)
## Running the Code
1. Clone the repository.
2. Make sure the data.csv file is in the same directory as the script.
3. Run the Python script:

```
python classification.py
```

## Output
The following will be printed to the console:
- Median activity threshold
- Class distribution
- Accuracy and classification reports for both LR and RF

The following plots will be displayed:
- Confusion matrix heatmaps for LR and RF
- Random Forest feature importance bar graph

A decision tree visualization from the random forest model will be generated in the same directory as the script.

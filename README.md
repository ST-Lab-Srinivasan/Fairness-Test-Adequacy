================================================================================
MUTATION-BASED FAIRNESS TEST ADEQUACY FOR MACHINE LEARNING SYSTEMS
===============================================================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20061830.svg)](https://doi.org/10.5281/zenodo.20061830)

Title: Mutation-Based Fairness Test Adequacy for Machine Learning Systems
Authors: Kehinde Akinola, Madhusudan Srinivasan
Institution: East Carolina University, Greenville, North Carolina, United States


================================================================================
DESCRIPTION
================================================================================

This repository contains the implementation code for evaluating fairness test 
adequacy in machine learning systems using mutation-based testing. The framework 
introduces 12 fairness-aware mutation operators that systematically inject 
controlled perturbations into training data to assess whether test datasets can 
effectively detect fairness violations.

The approach treats fairness as a non-functional software quality attribute and 
provides a quantifiable mutation score that indicates the fault-revealing 
capability of fairness test suites across three group fairness metrics:
- Demographic Parity (DP)
- Equal Opportunity (EO)
- Equalized Odds (EOd)

================================================================================
DATASET INFORMATION
================================================================================

The framework has been evaluated on five real-world datasets:

1. ADULT INCOME DATASET
   - Source: UCI Machine Learning Repository / Census Income
   - Instances: 48,842 (Training: 32,561 | Testing: 16,281)
   - Features: 14 demographic and socioeconomic attributes
   - Task: Binary classification (income >50K or <=50K)
   - Sensitive Attributes: sex, race, age, marital_status, education
   - URL: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

2. COMPAS DATASET
   - Source: ProPublica criminal justice records
   - Instances: 7,214
   - Features: 52 attributes (age, sex, race, prior offenses, COMPAS scores)
   - Task: Recidivism prediction (will reoffend in 2 years)
   - Sensitive Attributes: race, sex, age
   - URL: https://mlr3fairness.mlr-org.com/reference/compas.html

3. GERMAN CREDIT DATASET
   - Source: UCI Machine Learning Repository
   - Instances: 1,000
   - Features: 20 financial and personal attributes
   - Task: Credit risk classification (good/bad credit)
   - Sensitive Attributes: age, sex, employment status
   - URL: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

4. CREDIT CARD DEFAULT DATASET
   - Source: Taiwan credit card clients
   - Instances: 30,000
   - Features: 20+ attributes (demographics, payment history)
   - Task: Default prediction (next month payment default)
   - Sensitive Attributes: sex, education, age
   - URL: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

5. BANK MARKETING DATASET
   - Source: Portuguese bank marketing campaigns
   - Instances: 45,211
   - Features: 16+ attributes (age, job, marital status, education)
   - Task: Term deposit subscription prediction
   - Sensitive Attributes: age, marital_status, education
   - URL: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

Dataset Preparation:
- All datasets use stratified 70/30 train-test splits (or predefined splits)
- Label encoding: Binary classification (0/1)
- Missing values handled via median/mode imputation
- Categorical features one-hot encoded
- Numeric features standardized (zero mean, unit variance)

================================================================================
CODE INFORMATION
================================================================================

STRUCTURE:
├── main.py                              # RQ1: Mutation-based adequacy evaluation
├── get_rq2_graph.py                     # RQ1: Dataset-specific execution (COMPAS example)
├── result_compare_mutation_baseline.py  # RQ2: Baseline comparison
├── rq3_answer.py                    # RQ3: Test set strength analysis
├── generate_graph_baseline_comparison.py # RQ2: Visualization of baseline comparison
├── heat_map_result.py                   # RQ1: Heatmap visualization of mutation scores
└── README.txt                           # This file

--------------------------------------------------------------------------------
FILE 1: main.py - RESEARCH QUESTION 1 (RQ1)
--------------------------------------------------------------------------------

PURPOSE: 
Evaluates the effectiveness of mutation-based fairness test adequacy across 
different ML models, datasets, and mutation operators.

RESEARCH QUESTION:
"How effective is the proposed fairness test adequacy approach in detecting 
fairness violations across different ML models, datasets, and mutation operators?"

KEY COMPONENTS:

1. MUTATION OPERATORS (12 total):
   
   a) REMOVAL OPERATORS:
      - Instance Removal (binary, multivalue, combinatorial)
      - Feature Removal (single, multi-feature, combinatorial)
   
   b) PERMUTATION OPERATORS:
      - Feature Permutation
      - Row Permutation
      - Label Flip (group-specific, multi-group)
   
   c) NOISE INJECTION:
      - Attribute-wise Gaussian noise on sensitive features
   
   d) DISTRIBUTION DRIFT:
      - Sensitive-feature scaling drift
   
   e) GROUP-SPECIFIC CORRUPTION:
      - Subgroup-targeted numeric perturbations
   
   f) SAMPLING OPERATORS:
      - Minority Oversampling
      - Majority Undersampling
   
   g) COUNTERFACTUAL SWAP:
      - Sensitive attribute value swapping

2. FAIRNESS METRICS:
   - Demographic Parity (DP): Equal positive prediction rates
   - Equal Opportunity (EO): Equal true positive rates
   - Equalized Odds (EOd): Equal TPR and FPR across groups

3. MODEL ARCHITECTURES:
   - Logistic Regression (LR)
   - Decision Tree (DT)
   - Random Forest (RF)
   - Gradient Boosting (GB)
   - K-Nearest Neighbors (KNN)

4. MUTATION SCORE CALCULATION:
   - Threshold: mean + 1.5*std deviation of fairness changes
   - Status: Mutant "killed" if fairness deviation exceeds threshold
   - Score: Percentage of killed mutants

5. DISTRIBUTIONAL FILTERING:
   - Filters mutants with distribution shift > 5% to ensure realistic perturbations
   - Prevents confounding between fairness violations and data quality issues

OUTPUTS:
- detailed.csv: Per-mutant results for each model and metric
- master_summary.csv: Aggregated results across all models
- master_operator_summary.csv: Operator-level mutation scores

CONFIGURATION PARAMETERS:
- RANDOM_STATE: 42 (reproducibility)
- MIN_GROUP_SIZE: 10 (minimum instances per demographic group)
- HIGH_SHIFT_THRESHOLD: 5.0% (distribution similarity filter)
- MAX_MUTANTS: 5000 (computational feasibility cap)
- MAX_PER_OPERATOR: 20 (per-operator mutant limit)

--------------------------------------------------------------------------------
FILE 2: get_rq2_graph.py - DATASET-SPECIFIC EXECUTION (RQ1)
--------------------------------------------------------------------------------

PURPOSE:
A self-contained, dataset-specific implementation of the mutation-based fairness 
test adequacy framework. This script demonstrates the complete pipeline for the 
COMPAS dataset and serves as a template for running experiments on other datasets.

RESEARCH QUESTION SUPPORTED:
RQ1 - "How effective is the proposed fairness test adequacy approach in detecting 
fairness violations across different ML models, datasets, and mutation operators?"

KEY COMPONENTS:

1. DATASET CONFIGURATION:
   - Configured for COMPAS recidivism prediction dataset
   - Label column: "two_year_recid" (binary: will reoffend in 2 years)
   - Sensitive attributes: race, sex, age, age_cat, c_charge_degree, 
     priors_count, juv_fel_count, juv_misd_count, juv_other_count
   - Numeric candidates for mutation operators: age, priors_count, 
     decile_score, days_b_screening_arrest, juvenile counts

2. DATA LOADING AND PREPROCESSING:
   - load_compas(): Reads CSV, normalizes column names, validates label column
   - create_train_test(): Stratified 70/30 train-test split
   - prepare_xy(): Feature engineering pipeline including:
     * Numeric conversion with NaN/inf handling
     * Median imputation for missing values
     * 99th percentile clipping for outliers
     * One-hot encoding for categorical variables
     * Column alignment between train and test sets

3. FAIRNESS METRICS IMPLEMENTATION:
   - compute_equalized_odds(): Calculates TPR and FPR per demographic group
   - compute_equal_opportunity(): Calculates TPR per demographic group
   - compute_demographic_parity(): Calculates positive prediction rate per group
   - determine_status_per_groups(): Threshold-based mutant kill determination
     using mean + 1*std of fairness differences

4. DISTRIBUTION SHIFT SCORING:
   - compute_distribution_shift_score(): Measures data drift between original 
     and mutated training data
   - Numeric columns: Standardized mean difference
   - Categorical columns: L1 distance between distributions
   - Mutants exceeding HIGH_SHIFT_THRESHOLD (5.0%) are filtered

5. MODEL TRAINING AND EVALUATION:
   - Supports 5 ML classifiers:
     * Logistic Regression (max_iter=500, balanced class weights)
     * Decision Tree (max_depth=10)
     * K-Nearest Neighbors (n_neighbors=5)
     * Random Forest (n_estimators=100, max_depth=10)
     * Gradient Boosting (n_estimators=100, max_depth=5)
   - StandardScaler applied to all features
   - NaN safety checks at multiple stages

6. MUTATION OPERATOR TOGGLES:
   - ENABLE_INTERSECTIONAL_LABEL_FLIP: True/False
   - ENABLE_GROUP_CORRUPTION: True/False
   - ENABLE_COUNTERFACTUAL_SWAP: True/False

7. CORE PIPELINE (run_model_block function):
   - Prepares training and test data
   - Trains baseline model and generates predictions
   - Creates test set variants (full, stratified, demographic-removed)
   - Builds all mutants and filters by distribution shift
   - Evaluates each mutant on each test set for all three fairness metrics
   - Records killed/alive status per sensitive attribute

8. OUTPUT GENERATION (write_master_tables function):
   - Concatenates results across all models
   - Generates master_summary.csv with detailed per-mutant results
   - Generates master_operator_summary.csv with aggregated operator statistics

OUTPUTS:
- OUTPUT_ROOT/equalized_odds/[model_name]/detailed.csv
- OUTPUT_ROOT/demographic_parity/[model_name]/detailed.csv
- OUTPUT_ROOT/equal_opportunity/[model_name]/detailed.csv
- OUTPUT_ROOT/[metric]/master_summary.csv
- OUTPUT_ROOT/[metric]/master_operator_summary.csv

CONFIGURATION:
- COMPAS_FILE: Path to compas-scores-two-years.csv
- OUTPUT_ROOT: Directory for output files
- LABEL_COL: Target column name ("two_year_recid")
- SENSITIVE_ATTRIBUTES: List of protected attributes
- LINEAR_NUMERIC_CANDIDATES: Numeric columns for mutation operators
- MIN_GROUP_SIZE: Minimum 10 instances per demographic group
- RANDOM_STATE: 42 for reproducibility
- MAX_MUTANTS: 5000 global cap
- MAX_PER_OPERATOR: 20 mutants per operator
- HIGH_SHIFT_THRESHOLD: 5.0% distribution shift filter

USAGE:
1. Update COMPAS_FILE path to your dataset location
2. Update OUTPUT_ROOT to desired output directory
3. Run: python get_rq2_graph.py
4. Results saved in OUTPUT_ROOT organized by fairness metric

ADAPTATION FOR OTHER DATASETS:
To use this script with a different dataset:
1. Update file path and LABEL_COL to match your dataset
2. Modify SENSITIVE_ATTRIBUTES list for your protected attributes
3. Update LINEAR_NUMERIC_CANDIDATES for numeric columns
4. Adjust load function if CSV format differs

--------------------------------------------------------------------------------
FILE 3: result_compare_mutation_baseline.py - RESEARCH QUESTION 2 (RQ2)
--------------------------------------------------------------------------------

PURPOSE:
Compares mutation-based adequacy against six baseline adequacy measures using 
Spearman rank correlation with actual fairness fault detection.

RESEARCH QUESTION:
"How does the mutation-based approach perform compared to baseline approaches 
for fairness test adequacy?"

BASELINE ADEQUACY METRICS:

1. COMBINATORIAL COVERAGE
   - Measures intersectional demographic combinations present in test set
   - Formula: |covered_pairs| / |total_pairs|
   - Inspired by combinatorial fairness testing (Kitamura et al. 2022)

2. DISTRIBUTIONAL BALANCE
   - Jensen-Shannon similarity between test and full dataset distributions
   - Formula: 1 - JSD(P_test, P_full) for each sensitive attribute
   - Averaged across all sensitive attributes

3. DISTANCE TO TRAINING (DSA)
   - Adapted from Distance-based Surprise Adequacy (Kim et al. 2019)
   - Formula: Mean k-nearest neighbor distance in transformed feature space
   - Uses k=5 neighbors with Euclidean distance

4. INDIVIDUAL DISCRIMINATION INDEX (IDI)
   - Percentage of test instances with counterfactual prediction changes
   - Formula: |{x : exists x' in CF(x), f(x) != f(x')}| / |Test Set|
   - Inspired by ADF and THEMIS approaches

5. DECISION BOUNDARY COVERAGE
   - Fraction of test instances near decision threshold
   - Formula: |{x : |p(x) - 0.5| < epsilon}| / |Test Set|
   - Uses epsilon = 0.05 (5% margin around 0.5 threshold)

6. FEATURE SPACE COVERAGE
   - Coverage of sensitive attribute value ranges
   - Bins numeric attributes (5 bins) and measures category coverage
   - Averaged across all sensitive attributes

METHODOLOGY:
1. Load mutation results from master_summary.csv (RQ1 output)
2. Split mutants: 70% training (adequacy) / 30% testing (fault yield)
3. Compute mutation adequacy on training mutants per test set
4. Compute fault yield on testing mutants per test set
5. Compute all baseline metrics for each test set
6. Calculate Spearman rank correlation: metric vs. fault yield

STATISTICAL ANALYSIS:
- Bootstrap resampling (10,000 iterations)
- 95% confidence intervals (percentile method)
- Benjamini-Hochberg FDR correction for multiple testing
- Significance levels: *** p<0.001, ** p<0.01, * p<0.05

OUTPUTS:
- merged_data_ALL_TESTSETS.csv: All metrics for all test sets
- final_correlation_ALL_TESTSETS.csv: Correlation results ranked by rho

EXPECTED RESULTS:
- Mutation Score achieves highest correlation (rho = 0.20 to 1.00)
- Statistically significant in all 15 dataset-metric combinations
- Baselines show inconsistent performance (8-9/15 significant cases)

--------------------------------------------------------------------------------
FILE 4: get_rq3_answer.py - RESEARCH QUESTION 3 (RQ3)
--------------------------------------------------------------------------------

PURPOSE:
Analyzes how mutation-based fairness adequacy varies with test set strength 
to validate that stronger test sets achieve higher mutation scores.

RESEARCH QUESTION:
"How does the mutation-based fairness adequacy vary with test set strength?"

TEST SET VARIANTS:

1. STRATIFIED SUBSETS:
   - 40% subset: Weak adequacy (reduced coverage)
   - 60% subset: Medium adequacy (moderate coverage)
   - 80% subset: Strong adequacy (near-complete coverage)
   - Stratified by label to preserve class distribution

2. GROUP-REMOVED TEST SETS:
   - Remove all instances of one sensitive group value
   - Examples: no_sex=Male, no_race=White, no_age=25
   - Tests impact of missing demographic coverage

3. FULL TEST SET:
   - Complete 30% holdout from train/test split
   - Baseline for maximum adequacy

ANALYSIS:
- Aggregates mutation scores by test set type
- Computes mean mutation score per fairness metric
- Validates monotonic relationship: 40% < 60% < 80% < Full
- Demonstrates that removed sets < full sets

METHODOLOGY:
1. Load master_summary CSV files for each fairness metric:
   - master_summary_demographic_parity.csv
   - master_summary_equal_opportunity.csv
   - master_summary_equalized_odd.csv

2. Map test set names to strength categories:
   - "40" in name -> 40%
   - "60" in name -> 60%
   - "80" in name -> 80%
   - "no_" prefix -> Removed
   - "full" -> Full

3. Compute mean mutation score (%) by test strength
4. Export pivoted table (rows=strength, columns=metrics)

OUTPUTS:
- rq3_mutation_vs_test_strength_all_metrics.csv
- Console output: Formatted table with mutation scores

EXPECTED PATTERN:
- Monotonic increase: 40% < 60% < 80% <= Full
- Removed sets < Full (typically 6-15% lower)
- Equalized Odds shows steepest increase
- Validates adequacy metric meaningfulness

--------------------------------------------------------------------------------
FILE 5: generate_graph_baseline_comparison.py - RQ2 VISUALIZATION
--------------------------------------------------------------------------------

PURPOSE:
Generates a publication-quality grouped bar chart comparing Spearman correlation 
coefficients across all adequacy criteria for the three fairness metrics 
(Demographic Parity, Equal Opportunity, Equalized Odds).

RESEARCH QUESTION SUPPORTED:
RQ2 - "How does the mutation-based approach perform compared to baseline 
approaches for fairness test adequacy?"

FUNCTIONALITY:

1. DATA LOADING:
   - Loads correlation results from three CSV files:
     * final_correlation_ALL_TESTSETS_demographic_parity.csv
     * final_correlation_ALL_TESTSETS_equal_oppurtunity.csv
     * final_correlation_ALL_TESTSETS_equalized_odd.csv
   - Extracts "Adequacy Criterion" and "Spearman rho" columns from each file

2. DATA MERGING:
   - Merges all three datasets on "Adequacy Criterion" column
   - Renames Spearman correlation columns to distinguish metrics:
     * Spearman_rho_DP (Demographic Parity)
     * Spearman_rho_EO (Equal Opportunity)
     * Spearman_rho_EOd (Equalized Odds)

3. ORDERING:
   - Applies preferred display order for adequacy criteria:
     1. Mutation Based (our proposed approach)
     2. Combinatorial Coverage
     3. Distributional Balance
     4. Distance to Training
     5. Individual Discrimination Index
     6. Decision Boundary Coverage

4. VISUALIZATION:
   - Creates grouped bar chart with three bars per adequacy criterion
   - Bar width: 0.25 (three bars side-by-side)
   - Horizontal reference line at y=0 for visual clarity
   - X-axis: Adequacy criteria (rotated 45 degrees for readability)
   - Y-axis: Spearman correlation coefficient (rho)

OUTPUT:
- rq2_correlation_all_metrics.png: High-resolution grouped bar chart showing 
  correlation comparison across all fairness metrics

INTERPRETATION:
The generated figure visually demonstrates that mutation-based adequacy 
consistently achieves the highest positive correlation with fault detection 
across all three fairness metrics, while baseline approaches show variable 
and often weaker correlations. This provides visual support for the claim 
that mutation-based adequacy is a superior predictor of fairness fault 
detection capability.

USAGE:
1. Ensure correlation CSV files are in the working directory
2. Run: python generate_graph_baseline_comparison.py
3. Output saved as: rq2_correlation_all_metrics.png

DEPENDENCIES:
- pandas: Data loading and manipulation
- numpy: Numerical operations for bar positioning
- matplotlib: Visualization and figure generation

--------------------------------------------------------------------------------
FILE 6: heat_map_result.py - RQ1 VISUALIZATION
--------------------------------------------------------------------------------

PURPOSE:
Generates publication-quality heatmaps showing mutation scores (%) across 
all combinations of ML models and mutation operators for each fairness metric.

RESEARCH QUESTION SUPPORTED:
RQ1 - "How effective is the proposed fairness test adequacy approach in 
detecting fairness violations across different ML models, datasets, and 
mutation operators?"

FUNCTIONALITY:

1. DATA LOADING:
   - Loads master summary CSV files for each fairness metric:
     * master_summary_demographic_parity.csv
     * master_summary_equal_oppurtunity.csv
     * master_summary_equalized_odd.csv

2. AGGREGATION:
   - Groups data by (Model, Operator) combinations
   - Computes mean mutation score for each combination
   - Converts scores to percentage scale (0-100%)
   - Rounds values to one decimal place for display

3. PIVOT TABLE CREATION:
   - Creates a 2D matrix with:
     * Rows: ML model names (Decision Tree, Gradient Boosting, KNN, 
       Logistic Regression, Random Forest)
     * Columns: Mutation operator names (counterfactual_swap, drift, 
       group_corruption, instance_removal, etc.)
   - Sorts both rows and columns alphabetically
   - Fills missing values with 0 for visual consistency

4. HEATMAP VISUALIZATION:
   - Color scheme: "Blues" colormap (light to dark blue)
   - Value range: 0-100% (fixed scale for cross-metric comparability)
   - Cell annotations: Exact mutation score values displayed in each cell
   - Adaptive text color: White text for dark cells (>60% of max), 
     black text for light cells (improved readability)

5. MULTIPLE OUTPUT GENERATION:
   - Processes all three fairness metrics in a single run
   - Generates separate heatmap for each metric

OUTPUTS:
- heatmap_demographic_parity_fixed.png: Heatmap for Demographic Parity metric
- heatmap_equal_opportunity_fixed.png: Heatmap for Equal Opportunity metric
- heatmap_equalized_odds_fixed.png: Heatmap for Equalized Odds metric

INTERPRETATION:
The heatmaps provide a comprehensive visual summary of mutation score patterns:
- High scores (dark blue): Mutation operators that effectively reveal fairness 
  faults detectable by the test suite
- Low scores (light blue/white): Operators producing mutants that survive 
  (potential gaps in test coverage)
- Cross-model comparison: Identifies which models are more sensitive to 
  specific types of fairness perturbations
- Cross-operator comparison: Reveals which mutation operators are most 
  effective for each model type

USAGE:
1. Ensure master_summary CSV files are in the working directory
2. Run: python heat_map_result.py
3. Three heatmap PNG files are generated in the current directory

DEPENDENCIES:
- pandas: Data loading, grouping, and pivot table creation
- numpy: Numerical operations and array handling
- matplotlib: Heatmap visualization and figure generation

EXAMPLE OUTPUT STRUCTURE:
                    counterfactual  drift  group_corruption  instance_removal  ...
decision_tree            45.2       67.8       72.1              58.3          ...
gradient_boosting        52.1       71.4       78.9              63.7          ...
knn                      38.9       59.2       65.4              51.8          ...
logistic_regression      49.7       68.5       74.2              60.1          ...
random_forest            51.3       70.9       77.5              62.4          ...

================================================================================
USAGE INSTRUCTIONS
================================================================================

PREREQUISITES:
- Python 3.8+
- Required packages (see Requirements section below)
- Datasets downloaded and placed in appropriate directories

--------------------------------------------------------------------------------
STEP 1: PREPARE ENVIRONMENT
--------------------------------------------------------------------------------

1. Install dependencies:
   pip install -r requirements.txt

2. Download datasets (see Dataset Information section for URLs)

3. Update file paths in each script:
   - main.py: TRAIN_FILE, TEST_FILE, OUTPUT_ROOT
   - get_rq2_graph.py: COMPAS_FILE, OUTPUT_ROOT
   - result_compare_mutation_baseline.py: OUTPUT_ROOT, TRAIN_PATH, TEST_PATH
   - get_rq3_answer.py: FILES dictionary with CSV paths

--------------------------------------------------------------------------------
STEP 2: RUN RQ1 - MUTATION-BASED ADEQUACY EVALUATION
--------------------------------------------------------------------------------

PURPOSE: Generate mutation scores across models, operators, and fairness metrics

COMMAND (Generic):
python main.py

COMMAND (COMPAS-specific):
python get_rq2_graph.py

EXECUTION TIME: 
- Adult dataset: ~3.2 hours (48K instances)
- COMPAS dataset: ~0.9 hours (7K instances)
- German Credit: ~0.4 hours (1K instances)
- Hardware: 12-core Intel Xeon, 64GB RAM

CONFIGURATION:
- Edit TRAIN_FILE and TEST_FILE paths for your dataset
- Adjust SENSITIVE_ATTRIBUTES list to match dataset
- Configure mutation operators (ENABLE_* flags)
- Set computational limits:
  * MAX_MUTANTS: Total mutant cap
  * MAX_PER_OPERATOR: Per-operator limit
  * HIGH_SHIFT_THRESHOLD: Distribution filter (recommended: 5.0)

OUTPUTS (in OUTPUT_ROOT):
OUTPUT_ROOT/
├── equalized_odds/
│   ├── [model_name]/
│   │   └── detailed.csv
│   ├── master_summary.csv
│   └── master_operator_summary.csv
├── demographic_parity/
│   └── [same structure]
└── equal_opportunity/
    └── [same structure]

OUTPUT FILE SCHEMAS:

detailed.csv:
- Operator: Mutation operator name
- Mutant: Unique mutant identifier
- TestSet: Test set name (full, stratified_40, no_sex=Male, etc.)
- Adequacy: Test set strength category
- Killed: Number of sensitive attributes showing violations
- Alive: Number of sensitive attributes not showing violations
- MutationScore: Killed / (Killed + Alive)

master_summary.csv:
- All columns from detailed.csv
- Model: ML model name (decision_tree, logistic_regression, etc.)

master_operator_summary.csv:
- Operator: Mutation operator name
- TestSet: Test set name
- TotalMutants: Unique mutants for this operator
- TotalKilled: Cumulative killed count
- TotalAlive: Cumulative alive count
- MutationScore: Aggregated score

--------------------------------------------------------------------------------
STEP 3: RUN RQ2 - BASELINE COMPARISON
--------------------------------------------------------------------------------

PURPOSE: Compare mutation adequacy against 6 baseline metrics

PREREQUISITES:
- RQ1 must be completed (master_summary.csv required)

COMMAND:
python result_compare_mutation_baseline.py

EXECUTION TIME: ~30-45 minutes per dataset

CONFIGURATION:
- Update OUTPUT_ROOT to match RQ1 output directory
- Set TRAIN_PATH and TEST_PATH to original dataset paths
- Verify SENSITIVE_ATTRIBUTES matches dataset

OUTPUTS (in OUTPUT_ROOT):
- merged_data_ALL_TESTSETS.csv: All metrics for each test set
- final_correlation_ALL_TESTSETS.csv: Spearman correlations ranked
- model.pkl: Trained logistic regression model (for IDI/Boundary baselines)

INTERPRETATION:
- Spearman rho closer to 1.0 = stronger predictive power
- Statistical significance: *** p<0.001, ** p<0.01, * p<0.05
- 95% confidence intervals shown in brackets

EXPECTED OUTPUT FORMAT:
Adequacy Criterion               Category            Metric        Spearman rho  Valid Pairs
Mutation Score (OURS)           Fault-Based         MutationAdeq.    0.68***      107
Combinatorial Coverage          Coverage-Based      DatasetCov.      0.40***      107
Distributional Balance          Distribution-Based  BalanceToFull    0.31**       107
Distance to Training            Distance-Based      DSA             -0.30*        107
Individual Discrimination IDX   Counterfactual      IDI_Percentage  -0.17         107
Decision Boundary Coverage      Boundary-Based      BoundaryCov.     0.05         107

--------------------------------------------------------------------------------
STEP 4: RUN RQ2 VISUALIZATION - BASELINE COMPARISON GRAPH
--------------------------------------------------------------------------------

PURPOSE: Generate publication-quality bar chart comparing correlation coefficients

PREREQUISITES:
- RQ2 must be completed for all three fairness metrics
- Required files in working directory:
  * final_correlation_ALL_TESTSETS_demographic_parity.csv
  * final_correlation_ALL_TESTSETS_equal_oppurtunity.csv
  * final_correlation_ALL_TESTSETS_equalized_odd.csv

COMMAND:
python generate_graph_baseline_comparison.py

EXECUTION TIME: <1 minute (visualization only)

CONFIGURATION:
- Update file paths (dp_file, eo_file, eod_file) if using different naming
- Adjust preferred_order list to change bar ordering
- Modify figure dimensions if needed (default: 7.0 x 3.5 inches)

OUTPUTS:
- rq2_correlation_all_metrics.png: Grouped bar chart (600 DPI, PeerJ-compliant)

VISUALIZATION FEATURES:
- Three bars per adequacy criterion (DP, EO, EOd)
- Horizontal reference line at y=0
- Rotated x-axis labels for readability
- Legend identifying each fairness metric

--------------------------------------------------------------------------------
STEP 5: RUN RQ1 VISUALIZATION - MUTATION SCORE HEATMAPS
--------------------------------------------------------------------------------

PURPOSE: Generate publication-quality heatmaps showing mutation scores by 
model and operator combinations

PREREQUISITES:
- RQ1 must be completed for all three fairness metrics
- Required files in working directory:
  * master_summary_demographic_parity.csv
  * master_summary_equal_oppurtunity.csv
  * master_summary_equalized_odd.csv

COMMAND:
python heat_map_result.py

EXECUTION TIME: <1 minute (visualization only)

CONFIGURATION:
- Update paths dictionary if using different file naming
- Adjust figure dimensions if needed (default: 7.0 x 4.5 inches)
- Modify colormap if desired (default: "Blues")

OUTPUTS:
- heatmap_demographic_parity_fixed.png
- heatmap_equal_opportunity_fixed.png
- heatmap_equalized_odds_fixed.png

VISUALIZATION FEATURES:
- Color-coded cells (darker = higher mutation score)
- Exact values annotated in each cell
- Adaptive text color for readability
- Colorbar with percentage scale (0-100%)

--------------------------------------------------------------------------------
STEP 6: RUN RQ3 - TEST SET STRENGTH ANALYSIS
--------------------------------------------------------------------------------

PURPOSE: Validate monotonic relationship between test strength and adequacy

PREREQUISITES:
- RQ1 must be completed for all three fairness metrics

COMMAND:
python get_rq3_answer.py

EXECUTION TIME: <1 minute (data aggregation only)

CONFIGURATION:
- Update FILES dictionary with paths to master_summary CSVs:
  FILES = {
      "Demographic Parity": "path/to/master_summary_demographic_parity.csv",
      "Equal Opportunity": "path/to/master_summary_equal_oppurtunity.csv",
      "Equalized Odds": "path/to/master_summary_equalized_odd.csv"
  }
- Verify ORDER list matches your test set naming convention

OUTPUTS:
- rq3_mutation_vs_test_strength_all_metrics.csv: Pivoted table
- Console output: Formatted table

EXPECTED OUTPUT FORMAT:
RQ3: Mean Mutation Score (%) vs Test Set Strength

TestStrength  Demographic Parity  Equal Opportunity  Equalized Odds
40%                    51.99              51.83            56.97
60%                    60.23              59.85            65.56
80%                    64.78              64.37            70.05
Removed                61.64              61.10            69.55
Full                   63.99              64.66            78.49

KEY VALIDATION:
- Monotonic increase: 40% < 60% < 80% <= Full (check)
- Removed < Full in most cases (check)
- Equalized Odds highest scores (check)
- Improvement range: 6-15% from 40% to Full (check)

================================================================================
REQUIREMENTS
================================================================================

PYTHON VERSION:
- Python 3.8 or higher

CORE DEPENDENCIES:
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

ML LIBRARIES:
torch>=1.9.0                 # PyTorch (GPU support optional)
xgboost>=1.4.0              # Gradient Boosting

UTILITIES:
matplotlib>=3.4.0            # Visualization (required for heatmaps and graphs)
seaborn>=0.11.0             # Statistical plots (optional)

INSTALL ALL:
Create requirements.txt with above packages and run:
pip install -r requirements.txt

GPU SUPPORT (OPTIONAL):
The code includes GPU acceleration via PyTorch but defaults to CPU:
- USE_GPU = torch.cuda.is_available()
- DEVICE = torch.device("cuda" if USE_GPU else "cpu")
- If GPU unavailable, code runs on CPU automatically
- GPU recommended for large datasets (>30K instances) but not required

TESTED ENVIRONMENT:
- OS: Ubuntu 24.04 LTS / Windows 10+
- CPU: Intel Xeon E5-2680 v4 (12 cores)
- RAM: 64GB
- GPU: NVIDIA Tesla V100 (optional)

================================================================================
METHODOLOGY
================================================================================

MUTATION-BASED ADEQUACY FRAMEWORK:

1. FAULT INJECTION (Mutation Generation):
   - Apply 12 fairness-aware operators to training data
   - Each operator simulates realistic fairness failure modes:
     * Instance Removal: Sampling bias, missing data
     * Label Flip: Annotation errors, label bias
     * Drift: Temporal/demographic distribution shifts
     * Noise: Measurement errors
     * Group Corruption: Quality disparities

2. DISTRIBUTIONAL FILTERING:
   - Compute distribution shift score between original and mutated training data
   - Filter mutants where shift exceeds tau = 5%
   - Prevents unrealistic perturbations and confounding effects
   - Metrics: L1 distance (categorical), standardized mean difference (numeric)

3. MODEL RETRAINING:
   - Train model M' on mutated training data D'
   - Use same architecture and hyperparameters as baseline model M
   - Test both models on identical test set D_test

4. FAIRNESS EVALUATION:
   - Compute group fairness metrics on baseline and mutant predictions
   - For each sensitive group g and metric F:
     Delta_g = |F_g(M) - F_g(M')|
   
5. MUTANT CLASSIFICATION:
   - Detection threshold: theta = mean + k*std (k=1.5)
   - Mutant "killed" if max_g(Delta_g) > theta
   - Otherwise "alive"

6. ADEQUACY QUANTIFICATION:
   - Mutation Score = (# killed) / (# total valid mutants) x 100%
   - Higher scores indicate stronger fault-detection capability

STATISTICAL VALIDATION:
- Spearman rank correlation (non-parametric, robust to outliers)
- Bootstrap confidence intervals (10,000 iterations)
- Benjamini-Hochberg FDR correction (multiple testing)
- Repeated experiments with 10 random seeds {42-51}

THRESHOLD SELECTION (k=1.5):
- Distribution-free (Chebyshev's inequality)
- Consistent inflection points across datasets
- Discriminative power without saturation
- Cross-dataset stability (CV=8.2%)

================================================================================
INTERPRETATION GUIDE
================================================================================

MUTATION SCORE INTERPRETATION:

High-Stakes Domains (criminal justice, lending, healthcare):
-> Target: >=80% under Equalized Odds
-> Indicates: Strong fault-detection capability
-> Action: Deploy with confidence, monitor continuously

Medium-Stakes Domains (advertising, recommendations):
-> Target: 60-80% under Equal Opportunity or Equalized Odds
-> Indicates: Reasonable adequacy
-> Action: Deploy with monitoring, plan periodic re-evaluation

Lower-Stakes Domains (entertainment, non-personalized):
-> Target: 40-60% under Demographic Parity
-> Indicates: Baseline adequacy
-> Action: Deploy with awareness, investigate if violations detected

OPERATOR-LEVEL INSIGHTS:

High Kill Rates (70-100%):
- Instance-level operators (instance_removal, intersectional_instance)
- Distributional operators (drift, group_corruption, noise_injection)
-> Test suite effectively detects group representation issues

Medium Kill Rates (40-70%):
- Sampling operators (oversample, undersample)
- Label-based operators (label_flip)
-> Test suite partially adequate, consider augmentation

Low Kill Rates (<40%):
- Structural operators (counterfactual_swap, column_removal)
-> Test suite may lack adversarial or proxy-sensitive coverage
-> Consider targeted augmentation

FAIRNESS METRIC PATTERNS:

Equalized Odds > Equal Opportunity > Demographic Parity:
- This pattern indicates distributional operators dominate mutations
- EOd's dual constraints (TPR + FPR) detect more violations
- Typical relative improvement: 18-36%

CORRELATION INTERPRETATION (RQ2):

rho >= 0.7: Strong predictive power (Mutation Score typically achieves this)
0.5 <= rho < 0.7: Moderate predictive power
0.3 <= rho < 0.5: Weak predictive power
rho < 0.3: Poor predictive power (most baselines fall here)

Negative rho: Anti-correlated (e.g., Distance to Training)
-> Higher novelty -> lower fault detection

TEST STRENGTH VALIDATION (RQ3):

Expected Pattern: 40% < 60% < 80% <= Full
- Monotonic increase validates adequacy meaningfulness
- Non-monotonic patterns suggest:
  * Insufficient mutant diversity
  * Inadequate distributional filtering
  * Dataset-specific biases

Removed < Full:
- Confirms demographic coverage importance
- Typical gap: 6-15%
- Larger gaps indicate intersectional vulnerabilities

================================================================================
PRACTICAL GUIDANCE FOR PRACTITIONERS
================================================================================

GETTING STARTED:

1. QUICK EVALUATION (1-2 hours):
   - Use Tier 1 operators only (3 operators)
   - Single model (Gradient Boosting or Random Forest)
   - Single fairness metric (Equalized Odds)
   - 70% fault detection at 30% computational cost

2. STANDARD EVALUATION (4-6 hours):
   - Use Tier 1 + Tier 2 operators (6 operators)
   - 2-3 models (GB + LR + RF)
   - All three fairness metrics
   - 80% fault detection at 50% computational cost

3. COMPREHENSIVE EVALUATION (8-12 hours):
   - All 12 operators
   - All 5 models
   - All three fairness metrics
   - Full experimental protocol (as in paper)

RESPONDING TO LOW SCORES:

If Mutation Score < 60%:
1. Identify low-kill-rate operators (operator_summary.csv)
2. Augment test set to address gaps:
   - Low instance_removal -> Add minority subsamples
   - Low drift -> Include temporal/regional variants
   - Low intersectional -> Oversample intersections
3. Re-evaluate and verify improvement (target: +15-20%)

If Correlation < 0.5 (RQ2):
- Check dataset quality (missing values, encoding errors)
- Verify sensitive attributes are correctly identified
- Consider dataset-specific operator tuning

If Non-Monotonic Pattern (RQ3):
- Inspect distributional filtering logs
- Verify test set construction (stratification)
- Check for data leakage between train/test

PRODUCTION INTEGRATION:

Pre-Deployment:
- Run comprehensive evaluation (all operators)
- Require >=80% for high-stakes, >=60% for medium-stakes
- Document mutation scores in model cards

Continuous Monitoring:
- Quarterly re-evaluation (high-stakes)
- Semi-annual (medium-stakes)
- Use Tier 1+2 operators (50% time reduction)

CI/CD Integration:
- Set mutation score as deployment gate
- Parallelize across cores (sub-hour execution)
- Trigger re-evaluation on fairness metric degradation

COMPUTATIONAL OPTIMIZATION:

For Large Datasets (>50K instances):
1. Use MAX_PER_OPERATOR cap (e.g., 20)
2. Prioritize Tier 1 operators
3. Enable GPU acceleration (torch.cuda)
4. Parallelize mutant evaluations

For Limited Resources:
1. Stratified train/test split (80/20 instead of 70/30)
2. Single model evaluation (Gradient Boosting)
3. Focus on Equalized Odds only
4. Batch process operators sequentially

================================================================================
TROUBLESHOOTING
================================================================================

COMMON ISSUES:

1. MEMORY ERROR (MemoryError or OOM):
   Cause: Too many mutants or large feature space
   Solution:
   - Reduce MAX_MUTANTS (e.g., 5000 -> 2000)
   - Reduce MAX_PER_OPERATOR (e.g., 20 -> 10)
   - Process operators sequentially instead of batch
   - Enable ENABLE_* flags selectively

2. LOW MUTATION SCORES (<30%):
   Cause: Over-conservative threshold or robust model
   Solution:
   - Verify distribution filtering (HIGH_SHIFT_THRESHOLD)
   - Check operator diversity (operator_summary.csv)
   - Reduce k threshold (1.5 -> 1.0 for sensitivity analysis)
   - Inspect killed vs. alive breakdown by operator

3. ALL MUTANTS ALIVE (Score = 0%):
   Cause: Under-sensitive threshold or model invariance
   Solution:
   - Verify test set has sufficient sensitive groups (MIN_GROUP_SIZE)
   - Check if model uses sensitive attributes (SHAP/feature importance)
   - Examine fairness metric distributions (may be perfectly fair)
   - Lower k threshold experimentally (1.5 -> 0.5)

4. DISTRIBUTION SHIFT TOO HIGH (All mutants filtered):
   Cause: Aggressive operators or small dataset
   Solution:
   - Increase HIGH_SHIFT_THRESHOLD (5.0 -> 7.5)
   - Disable heavy operators (ENABLE_COUNTERFACTUAL_SWAP = False)
   - Use stratified sampling in operators
   - Check operator implementation for bugs

5. CONVERGENCE WARNINGS (ML models):
   Cause: Mutated data distribution differs from original
   Solution:
   - Increase model max_iter (e.g., LogisticRegression(max_iter=2000))
   - Add early stopping for iterative models
   - Use StandardScaler on all numeric features
   - Clip extreme values in mutated data

6. INCONSISTENT RESULTS ACROSS RUNS:
   Cause: Stochastic models or insufficient seeds
   Solution:
   - Set RANDOM_STATE consistently (42)
   - Increase random seed range (10 seeds -> 20 seeds)
   - Use deterministic training modes where available
   - Report mean +/- std across multiple runs

7. MISMATCHED TEST SET NAMES (RQ3):
   Cause: Naming convention mismatch
   Solution:
   - Inspect master_summary.csv TestSet column
   - Update map_test_strength() function in get_rq3_answer.py
   - Ensure consistent naming: "stratified_40", "no_sex=Male", "full"

8. BASELINE CORRELATION ERRORS (RQ2):
   Cause: Missing test sets or insufficient variance
   Solution:
   - Verify test_sets dictionary covers ALL master_summary test sets
   - Check for test sets with <10 instances (filtered out)
   - Ensure at least 3 test sets per adequacy level for correlation

9. GPU OUT OF MEMORY:
   Cause: Large batch size or model complexity
   Solution:
   - Disable GPU: USE_GPU = False
   - Reduce batch size in DataLoader
   - Use CPU-only execution (default fallback)

10. IMPORT ERRORS:
    Cause: Missing dependencies
    Solution:
    - Install all requirements: pip install -r requirements.txt
    - Verify Python version: python --version (>=3.8)
    - Check package versions: pip list

11. VISUALIZATION FILE NOT FOUND ERRORS:
    Cause: CSV files not in expected location for graph/heatmap scripts
    Solution:
    - Verify all correlation CSV files exist before running 
      generate_graph_baseline_comparison.py
    - Verify all master_summary CSV files exist before running 
      heat_map_result.py
    - Update file paths in the scripts to match your directory structure
    - Ensure file names match exactly (check for typos like 
      "oppurtunity" vs "opportunity")

12. HEATMAP DISPLAY ISSUES:
    Cause: Missing data or incompatible column names
    Solution:
    - Verify master_summary.csv contains "Model", "Operator", and 
      "MutationScore" columns
    - Check for NaN values in MutationScore column
    - Ensure at least one row exists per Model-Operator combination

================================================================================
CITATIONS
================================================================================

If you use this code or methodology in your research, please cite:

@article{akinola2026mutation,
  title={Mutation-Based Fairness Test Adequacy for Machine Learning Systems},
  author={Akinola, Kehinde and Srinivasan, Madhusudan},
  journal={PeerJ Computer Science},
  year={2026},
  note={Under review}
}

Related Work:

[1] Mutation Testing:
Jia, Y., & Harman, M. (2010). An analysis and survey of the development of 
mutation testing. IEEE Transactions on Software Engineering, 37(5), 649-678.

[2] Group Fairness Metrics:
Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in 
supervised learning. Advances in Neural Information Processing Systems, 29.

[3] Fairness Testing:
Galhotra, S., Brun, Y., & Meliou, A. (2017). Fairness testing: Testing 
software for discrimination. ESEC/FSE 2017, pp. 498-510.

[4] ML Testing Adequacy:
Kim, J., Feldt, R., & Yoo, S. (2019). Guiding deep learning system testing 
using surprise adequacy. ICSE 2019, pp. 1039-1049.

[5] Bias in ML Systems:
Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). 
A survey on bias and fairness in machine learning. ACM Computing Surveys, 
54(6), 1-35.

================================================================================
LICENSE & CONTRIBUTION GUIDELINES
================================================================================

LICENSE:
This code is released under the MIT License for academic and research purposes.

Copyright (c) 2026 Kehinde Akinola, Madhusudan Srinivasan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CONTRIBUTION GUIDELINES:

We welcome contributions to improve this framework! Areas of interest include:

1. NEW MUTATION OPERATORS:
   - Causal fairness operators
   - Temporal fairness perturbations
   - Multi-label fairness mutations
   - Context-dependent fairness operators

2. ADDITIONAL FAIRNESS METRICS:
   - Individual fairness (counterfactual fairness)
   - Causal fairness (path-specific effects)
   - Intersectional fairness metrics
   - Long-term fairness metrics

3. EXTENDED MODEL SUPPORT:
   - Deep neural networks (CNN, RNN, Transformers)
   - Multi-modal models (vision + language)
   - Federated learning systems
   - Online learning models

4. PERFORMANCE OPTIMIZATIONS:
   - Distributed mutant evaluation
   - Incremental mutation testing
   - Approximation techniques for large-scale systems
   - GPU acceleration improvements

5. TOOLING ENHANCEMENTS:
   - Web-based dashboard
   - Automated report generation
   - CI/CD integration templates
   - Docker containerization

To contribute:
1. Fork the repository
2. Create a feature branch (git checkout -b feature/YourFeature)
3. Commit your changes with clear messages
4. Add unit tests for new functionality
5. Update documentation (README and code comments)
6. Submit a pull request with detailed description

CONTACT:
For questions, bug reports, or collaboration inquiries:
- Madhusudan Srinivasan: srinivasanm23@ecu.edu
- Kehinde Akinola: akinolak23@ecu.edu

ACKNOWLEDGMENTS:
This work was conducted at the Department of Computer Science, 
East Carolina University.

================================================================================
ADDITIONAL NOTES
================================================================================

REPRODUCIBILITY:
- All experiments use fixed random seeds (42-51)
- Results averaged across 10 independent runs
- Statistical significance assessed with bootstrap methods
- Dataset splits and preprocessing scripts provided

COMPUTATIONAL RESOURCES:
- Standard workstation sufficient (12-core CPU, 64GB RAM)
- GPU optional (reduces time by ~40% for large datasets)
- Cloud execution supported (AWS, Google Cloud, Azure)

EXTENSIONS:
- Framework modular and extensible
- Easy to add custom mutation operators (inherit from base class)
- Custom fairness metrics supported (implement compute_* functions)
- Dataset-agnostic (works with any tabular classification dataset)

KNOWN LIMITATIONS:
- Focus on binary classification (multi-class extension in progress)
- Tabular data only (image/text extensions planned)
- Post-processing fairness evaluation (in-training not supported)
- Assumes static models (concept drift monitoring separate concern)

FUTURE WORK:
- Integration with MLOps platforms (MLflow, Kubeflow)
- Real-time fairness monitoring dashboard
- Automated remediation suggestions
- Causal fairness operator extensions
- Multi-objective adequacy optimization

================================================================================
VERSION HISTORY
================================================================================

Version 1.0 (January 2026):
- Initial release
- Support for 5 datasets, 5 models, 3 fairness metrics
- 12 mutation operators
- 6 baseline comparisons
- Complete RQ1-RQ3 experimental protocol

Planned Version 1.1:
- Deep learning model support (PyTorch, TensorFlow)
- Multi-class classification support
- Automated report generation
- Docker containerization
- Web UI dashboard

================================================================================
END OF README
================================================================================

For the most up-to-date information, documentation, and examples, please visit:
[Repository URL to be added upon publication]

Last Updated: January 2026
Document Version: 1.0
================================================================================

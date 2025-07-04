# WAE - evolutionary algorithm
Simple cma-es implementation with different mean population strategies 

## Requirements
python 3.8+

## Instalation

1. Clone/download the repository

2. Create and activate virtual environment: 
    python -m venv venv
    On Windows: venv\Scripts\activate
    On Linux/Mac: source venv/bin/activate

3. Install the required packages:
    pip install -r requirements.txt

## Usage
Usage: python run_all.py [flags]
Flags: e = experiments, w = wilcoxon, v = visualization, s - summary stats
experiments -> run performance and convergence experiments
wilcoxon ->  run Wilcoxon statistical tests
visualizations -> generate convergence plots
All results will be save in results folder

## Test reproduction 
To reproduce all tests and analyses, run the script with all flags enabled
-- python run_all.py ewvs
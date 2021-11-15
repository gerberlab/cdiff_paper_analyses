# Uncovering metabolomic and microbial biomarkers to predict CDI recurrence
____
This repository can be used to repeat the analyses and reproduce figures published in *Gut metabolites predict Clostridioides difficile recurrence* (link)  

To repeat all analyses by scratch, re-run predictive analyses.  This analysis will take time and it is recommended you use a cluster to take advantage of parallel computing.

To repeat all other analyses and reproduce figures, download results of predictive analysis from Zotero (link) and skip to section 2, Data analyses

## 1. Predictive analysis
____
### 1.1 Run individual predictive analysis scripts
Predictive analysis scripts are **logistic_regression.py**, **random_forest.py**, and **cox_regression.py**

These scripts can be run from the command line on any data source at any week, and can run the model with a held-out sample (i.e. to calculate the AUC/CI score with bootstrapping) or can run the model with all samples (i.e. to calculate the coefficients with bootstrapping)

**The options that must be specified are:**
- **-o**: output data path where pickle files of model results will be deposited  
- **-i**: data source; options are:
  - "metabs" for LC/MS untargeted metabolites
  - "scfa" for targeted short chain fatty acids
  - "16s" for 16S rRNA amplicon sequencing data (i.e. ASVs)
  - "demo" for clinical/demographic variables
  - Multiple data sources, where any of the above options are combinded with an underscore 
    - i.e. -i 'metabs_scfa' will run a model with both metabolite and SCFA data  
- **-week**: the week predictions will be made from; options are:
  - 0 for pre-treatment
  - 1 for week 1
  - 2 for week 2  
- **-ix**: sample to hold out (if -type == 'auc')  
- **-type** whether to hold out the sample number specified by -ix in order to calculate an AUC/CI on an N-1 data set (useful to obtain a variance over AUC/CI scores); options are:
  - 'auc' (to hold out the sample specified by -ix)
  - 'coef' (to not hold out any data)  

**Ex: run random forest on metabolites and SCFA data at week 1 with the fourth sample held out:**  
`
python3 ./random_forest.py -o 'output' -i 'metabs_SCFA' -week 1 -type 'auc' -ix 4 ` 

**Ex: run cox regression on 16s sequencing data and clinical/demographic data at pre-treatment 
with no samples held out:**  
`python3 ./cox_regression.py -o 'output' -i '16s_demo' -week 0 -type 'coef' -ix 4`  

### 1.2 Run N-fold predictive analysis with multiple models and/or multiple data sources in parallel 
To run multiple held-out samples at once and/or multiple data sources, use **dispatcher.py** (if running on a remote server 
with LSF job submission) or **dispatcher_local.py** (if running on a local computer)  

For either **dispatcher.py** or **dispatcher_local.py**, all held-out indices will be ran in parallel
*(note: weeks with less than 48 samples of data will error on higher indices, but will continue running all possible indices)*  
**The options to specify for these scripts are: **
- **-o**: output data path; if not specified, folder 'PredictiveAnalysisResults' will be created for output files in the current working directory
- **-models**: which models to run. Accepts multiple arguments. Options are:
  - 'LR' for logistic regression
  - 'cox' for cox regression
  - 'RF' for random forest
- **-i**: data source; options are the same as above, but accepts multiple options separated by a space
- **-weeks**: options the same as above, but accepts multiple options separated by a space
 
Running dispatcher.py or dispatcher_local.py will result in pickle files with data from each held-out run in the output folder (with separate folders for each week and model, and sub-folders for each data source ran)  

After running predictive analysis, medians and intervals for scores and feature coefficients/importances can be 
computed with the jupyter notebook **'Predictive Analysis Results Compilation and Fig 4 Creation.ipynb'**  

## 2. Data analyses
___
### All other data analyses in the paper can be run with the provided jupyter notebooks.
- **1. Alpha & Beta Diversity, Figure 3.ipynb**: 
  - Calculates alpha & beta diversity on the 16S rRNA amplicon sequencing data (ASVs) & performs accompaning statistical tests 
  - Calculates spearman PCoA dissimilarity on the metabolites and the accompaning statistical tests
  - Creates figure 2
  - Creates supplemental figures 1 and 2
- **2. Univariate analysis.ipynb**: 
  - Performs univariate analysis on the 16S data, metabolites, SCFAs, toxin/culture data, and clinical data 
- **3. Figure 3 - Heatmaps.ipynb**:
  - creates the heatmaps for figure 2
  - Run after **2. Univariate analysis.ipynb**
- **4.Enrichment Analysis.ipynb**:
  - Performs enrichment analysis for the metabolites and ASVs found to be significant in univariate analysis
  - Run after **2. Univariate analysis.ipynb**
- **5. Predictive Analysis Results Compilation and Fig 4 Creation.ipynb**: 
  - Compiles summary scores of the predictive analysis runs if left-out bootstrapping was performed
  - Creates figure 4  
  - Run after predictive analysis (section 1) or after downloading predictive analysis results from Zotero (link)
- **6. Figure 5 - Metabolite summary figure**: 
  - Makes figure 5
  - Run after notebooks: **2. Univariate analysis.ipynb** and **5. Predictive Analysis Results Compilation and Fig 4 Creation.ipynb**:


For best results, run notebooks in order. Notebooks 2 and 5 create files that are used in subsequent notebooks.  

In all notebooks, be sure to change the directory in the first code box to your home directory 
## Packages & versions
### R Packages & versions
R 4.1.0  
RStudio 1.3.959  
DESeq2 1.32.0  

### Python Packages and versions
appnope                   0.1.2            
argon2-cffi               20.1.0            
async_generator           1.10              
attrs                     21.2.0  
backcall                  0.2.0  
biopython                 1.79  
blas                      1.0  
bleach                    3.3.1    
brotlipy                  0.7.0            
ca-certificates           2021.5.30            
cachecontrol              0.12.6                     
certifi                   2021.5.30        
cffi                      1.14.6           
chardet                   4.0.0           
cloudpickle               2.0.0              
colorama                  0.4.4               
cryptography              3.4.7            
cycler                    0.10.0                   
cython                    0.29.24          
decorator                 5.0.9              
defusedxml                0.7.1             
ecos                      2.0.7.post1      
entrypoints               0.3              
et_xmlfile                1.1.0             
freetype                  2.10.4               
future                    0.18.2                     
hdmedians                 0.14.2             
idna                      2.10                
importlib-metadata        4.6.1              
importlib_metadata        1.5.0                      
iniconfig                 1.1.1               
intel-openmp              2021.2.0             
ipykernel                 5.3.4             
ipython                   7.22.0            
ipython_genutils          0.2.0               
jdcal                     1.4.1                      
jedi                      0.17.0                     
jinja2                    3.0.1              
joblib                    1.0.1               
jpeg                      9b                     
jsonschema                3.2.0                
jupyter_client            6.1.12              
jupyter_core              4.7.1            
jupyterlab_pygments       0.1.2              
kiwisolver                1.3.1            
lcms2                     2.12                 
libcxx                    12.0.1               
libffi                    3.3                  
libgfortran               3.0.1               
libllvm10                 10.0.1                
libpng                    1.6.37               
libsodium                 1.0.18               
libtiff                   4.2.0                
libwebp-base              1.2.0                
llvm-openmp               10.0.0                
llvmlite                  0.36.0           
lockfile                  0.12.2           
lz4-c                     1.9.3                
markupsafe                2.0.1             
matplotlib                3.3.4            
matplotlib-base           3.3.4             
matplotlib-venn           0.11.6                   
mistune                   0.8.4             
mkl                       2021.2.0          
mkl-service               2.3.0            
mkl_fft                   1.3.0            
mkl_random                1.2.1            
more-itertools            8.8.0               
msgpack-python            1.0.2            
natsort                   7.1.1             
nbclient                  0.5.3               
nbconvert                 6.1.0            
nbformat                  5.1.3              
ncurses                   6.2                 
nest-asyncio              1.5.1              
networkx                  2.6.2                    
notebook                  6.4.0              
numba                     0.53.1           
numexpr                   2.7.3            
numpy                     1.20.2           
numpy-base                1.20.2            
olefile                   0.46                     
openpyxl                  3.0.7              
openssl                   1.1.1l              
osqp                      0.5.0            
packaging                 21.0             
pandas                    1.2.5           
pandoc                    2.14.1             
pandocfilters             1.4.2                      
parso                     0.8.2              
patsy                     0.5.1                  
pexpect                   4.8.0             
pickleshare               0.7.5           
pillow                    8.2.0            
pip                       21.1.2           
pluggy                    0.13.1           
prometheus_client         0.11.0               
prompt-toolkit            3.0.17             
ptyprocess                0.7.0               
py                        1.10.0            
pycparser                 2.20               
pygments                  2.9.0              
pyopenssl                 20.0.1              
pyparsing                 2.4.7             
pyrsistent                0.17.3          
pysocks                   1.7.1             
pytest                    6.2.4           
python                    3.7.10               
python-dateutil           2.8.1              
python_abi                3.7                   
pytz                      2021.1             
pyzmq                     20.0.0             
readline                  8.1                  
requests                  2.25.1               
scikit-bio                0.5.6             
scikit-learn              0.24.2             
scikit-survival           0.15.0.post0     
scipy                     1.6.2             
seaborn                   0.11.1                    
send2trash                1.7.1              
setuptools                52.0.0             
shap                      0.39.0              
six                       1.16.0               
slicer                    0.0.7              
sqlite                    3.35.4               
statsmodels               0.12.2            
tbb                       2020.2               
terminado                 0.10.1           
testpath                  0.5.0              
threadpoolctl             2.1.0               
tk                        8.6.10                
toml                      0.10.2              
tornado                   6.1               
tqdm                      4.62.3              
traitlets                 5.0.5              
typing_extensions         3.10.0.0           
urllib3                   1.26.6             
wcwidth                   0.2.5                      
webencodings              0.5.1                     
wheel                     0.36.2             
xlrd                      2.0.1                
xz                        5.2.5                
zeromq                    4.3.4                 
zipp                      3.5.0               
zlib                      1.2.11               
zstd                      1.4.9     
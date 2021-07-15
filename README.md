# Representation Learning Methods for Stock Price Movement Prediction
> Master dissertation code of Thijs Vemeire <br>
> To obtain the degree of: Master of Science in Business Engineering, Data Analytics 

## Part 1: Data Collection 
All files and scripts used to collect the data for this work. 
A Batch file was created to run daily by windows task scheduler at the end of each trading day. 
The batch file includes the path to the python.exe directory and python script. 
The python script makes use of packages as YahooFinance, BeautifulSoup and Tweety to retrieve the information. 

- [Web Scraping](scraping.py)
- [Batch file](automated_scraping.bat)

## Part 2: Data Preprocessing 
For each analysis form a different notebook is created where basic data exploration and cleaning is done. 
 - [Fundamental data preprocessing](ppFundamental.ipynb)
 - [Technical data preprocessing](ppTechnical.ipynb)
 - [Text based data preprocessing](ppNaturalLanguage.ipynb)
 
The following [script](preprocessing.py) contains all text preprocessing steps needed to create a basetable that will be used in the following steps. 
 
 ## Part 3: Modelling 
 We test the following RL methods: PCA, KPCA, ICA, RBM, AE, DNN (LSTM) and CNN. For each of these a separate notebook can be found with the implementation.
 
 - [Notebook of CNN](CNN.ipynb)
 - [Notebook of PCA](PCA.ipynb)
 - [Notebook of KPCA](KPCA.ipynb)
 - [Notebook of ICA](ICA.ipynb)
 - [Notebook of AE](AE.ipynb)
 - [Notebook of RBM](RBM.ipynb)
 - [Notebook of DNN](LSTM.ipynb)

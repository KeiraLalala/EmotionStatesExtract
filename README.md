# EmotionStatesExtract
This project is used to extract the directory of onset, apex, and offset frame in each video, which should be the sample of micro-expression dataset. The label of 'onset, apex and offset' is marked by two .xlsx files, **../SAMM/Raw_Data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx**, and **../CASME-II/CASME2-coding-20190701.xlsx**, which could be found in CASMEII and SAMM datasets. Fo eSMIC, we extract the firt, middle and last frames. 

## File instruction

**EmoStatsSave.ipynb** extract and save the frame directory with .txt file

**EmoStatsLoad.ipynb** loading the files

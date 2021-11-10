# Hyb4mC
This is our implementation for the paper:
>Hyb4mC: a hybrid model based on DNA2vec for DNA N4-methylcytosine sites prediction
# Datasets
Due to size limitation, we put the paper data on github,available from https://github.com/YingLiangjxau/Hyb4mC.
# File Description
embedding_matrix.npy  
>The weight of the embedding layer converted from the pre-trained DNA vector provided by Ng (2017).  

dataprocess.py  
>Used for pre-processing DNA sequences.  

Hyb_Caps.py  
>The subnet constructed for small sample species can be used to train species-specific models and evaluate model performance.  

Hyb_Conv.py  
>The subnet constructed for large sample species can be used to train species-specific models and evaluate model performance.
# Usage
Enviroment:
>keras>=2.1.6  
tensorflow>=1.12.0  
python>=3.6  

Optional:  
>-name You can select the corresponding species from the following list :  ['A.thaliana', 'C.elegans', 'D.melanogaster', 'E.coli', 'G.subterraneus', 'G.pickeringii'].The default value is 0, and the corresponding species is 'A.thaliana'.

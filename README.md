# DeepPlanck
Deep learning mass estimates for Planck PLSZ2 clusters. The file containing the masses is DeepPlanck.csv. The file containes 4 columns: cluster, Planck_mass, CNN_mass and redsfhit. The cluster column corresponds to the name of the cluster in the PLSZ2 catalog. You can read the csv file using, for instance, pandas in Python. 

```
import pandas as pd
pd.read_csv('DeepPlanck.csv')
```
IMPORTANT: masses are given as the decimal logarithm of the mass in solar masses.

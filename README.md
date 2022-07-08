# uav_coverage

## Base requirements
It is recommended to use a python virtual environment such as virtualenv before installing the requirements.

Python 3.7

## Setup
To install the requirements
```pip install -r requirements.txt```

The custom environment must be registered to be used. To register the custom environment 'uav-v0'
Change directory into the parent uav_gym directory
```pip install -e .```


## Train
Call the train script with the env id, the number of UAVs, the coverage range, the percentage of users prioritised, and the prioritisation factor.
`bash train.sh uav-v0 <n_uavs> <cov_range> <pref_prop> <pref_fac>`

## Test
Call the test script with the env id, the number of UAVs, the coverage range, the percentage of users prioritised, and the prioritisation factor.

`bash test.sh uav-v0 <n_uavs> <cov_range> <pref_prop> <pref_fac>`

The results will be saved to the experiments directory.

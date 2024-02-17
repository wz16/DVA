###  DVA

This is the official implementation of the paper "One step closer to unbiased aleatoric uncertainty estimation" in AAAI-24. 

This code uses the age prediction model from [Yusuke Uchida](https://github.com/yu4u/age-estimation-pytorch) as regression model, to estimate the age data uncertainty.


###  Demo

After setting up the requirements and train the prediction model following [Yusuke Uchida](https://github.com/yu4u/age-estimation-pytorch),

train DVA uncertainty:
```
python train_with_uncertainty.py --uncertainty_load_prediction_resume [PATH/TO/TRAINED_CHECKPOINT] --s3y_mode heter --s2_mode heter
```

train VA uncertainty:
```
python train_with_uncertainty.py --uncertainty_load_prediction_resume [PATH/TO/TRAINED_CHECKPOINT] --s3y_mode fixed --s2_mode heter
```

drawing_std_from_saved.py is used to visualize the uncertainties.
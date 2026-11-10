"""The frozen pipeline: training-median imputation followed by native XGBoost."""
import hashlib
import numpy as np
import xgboost as xgb
from model_25 import METADATA, model_bytes

def load_model():
    raw=model_bytes()
    if hashlib.sha256(raw).hexdigest()!=METADATA['booster_json_sha256']:
        raise ValueError('Model checksum mismatch')
    booster=xgb.Booster(params={'nthread':1})
    booster.load_model(bytearray(raw))
    if booster.num_features()!=25:
        raise ValueError('Expected the frozen 25-feature model')
    return booster

def predict(booster, values):
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(25,): raise ValueError('Expected 25 values')
    if np.isinf(x).any(): raise ValueError('Infinite inputs are not allowed')
    missing=np.isnan(x)
    if missing.all(): raise ValueError('Enter at least one observed value or load the example.')
    filled=np.where(missing,np.asarray(METADATA['medians']),x)
    p=float(booster.predict(xgb.DMatrix(filled.reshape(1,-1),feature_names=booster.feature_names))[0])
    return {'probability':p,'positive':p>=METADATA['threshold'],'observed':int((~missing).sum()),'imputed':int(missing.sum())}

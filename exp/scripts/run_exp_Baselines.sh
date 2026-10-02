# Section 6.1 / Tables 2-4, 8: wrench estimation and time-series forecasting baselines
# Point-to-point estimators
python exp/scripts/baseline/exp_scratch_MINN.py
python exp/scripts/baseline/exp_scratch_RBF.py
python exp/scripts/baseline/exp_scratch_GPR.py

# Sequence-to-point estimators
python exp/scripts/baseline/exp_scratch_LSTM.py
python exp/scripts/baseline/exp_scratch_CNN.py

# Sequence-to-sequence estimators
python exp/scripts/baseline/exp_scratch_LSTMEncDec.py
python exp/scripts/baseline/exp_scratch_TransformerEncDec.py
python exp/scripts/baseline/exp_scratch_PatchTST.py
python exp/scripts/baseline/exp_scratch_PatchTST_Gaussian.py
python exp/scripts/baseline/exp_scratch_iTransformer.py

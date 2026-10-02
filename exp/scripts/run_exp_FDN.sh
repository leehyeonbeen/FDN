# Section 6.1: FDN trained from scratch (relative position inputs, Eq. 16)
python exp/scripts/fdn_relpos/exp_scratch_FDN_RelPos6D.py

# Section 6.2 / Table 5: architectural ablations
python exp/scripts/fdn_relpos/ablation/exp_ablation_FEF.py            # w/o FEF
python exp/scripts/fdn_relpos/ablation/exp_ablation_FEF_Weighting.py  # w/o FEF-W
python exp/scripts/fdn_relpos/ablation/exp_ablation_FEF_MOE.py        # w/o FEF-MoE
python exp/scripts/fdn_relpos/ablation/exp_ablation_FPF.py            # w/o FPF
python exp/scripts/fdn_relpos/ablation/exp_ablation_ModSpecEnc.py     # w/o ModSpec
python exp/scripts/fdn_relpos/ablation/exp_ablation_DetHead.py        # w/o TrdHead
python exp/scripts/fdn_relpos/ablation/exp_ablation_ProbHead.py       # w/o ResHead

# Section 6.3 / Table 6: input ablations
python exp/scripts/fdn_relpos/input_ablation/exp_input_ablation_NoJointVel.py     # w/o dq
python exp/scripts/fdn_relpos/input_ablation/exp_input_ablation_NoJointAcc.py     # w/o ddq
python exp/scripts/fdn_relpos/input_ablation/exp_input_ablation_NoJointVelAcc.py  # w/o dq, ddq
python exp/scripts/fdn_relpos/input_ablation/exp_input_ablation_NoJTorque.py      # w/o u (=dp)
python exp/scripts/fdn_relpos/input_ablation/exp_input_ablation_JointPosOnly.py   # w/o dq, ddq, u

# Section 6.4 / Table 7: residual correlation structures
python exp/scripts/fdn_relpos/residual_correlation/exp_residual_MVNChannel.py   # I_T x R_C
python exp/scripts/fdn_relpos/residual_correlation/exp_residual_MVNTemporal.py  # R_T x I_C
python exp/scripts/fdn_relpos/residual_correlation/exp_residual_MVNKron.py      # R_T x R_C

# Section 6.6 / Table 9: transfer learning from RH20T (R: relative position inputs)
python exp/scripts/fdn_relpos/pretrain/exp_pretrain.py
python exp/scripts/fdn_relpos/pretrain/exp_linprobe.py
python exp/scripts/fdn_relpos/pretrain/exp_finetune.py

# Section 6.6 / Table 9: transfer learning from RH20T (A: absolute position inputs)
python exp/scripts/fdn_abspos/exp_scratch_FDN_AbsPos6D.py
python exp/scripts/fdn_abspos/pretrain/exp_pretrain.py
python exp/scripts/fdn_abspos/pretrain/exp_linprobe.py
python exp/scripts/fdn_abspos/pretrain/exp_finetune.py

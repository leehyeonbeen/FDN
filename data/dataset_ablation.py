import torch

from data.dataset import HydraulicDatasetRelPos6D


# Input-ablation datasets: FDN inputs [dq, qdot, qddot, u, q0] x 6 joints with selected modalities removed
class _HydraulicDatasetRelPos6DInputAblation(HydraulicDatasetRelPos6D):
    input_indices = None

    def declare_index_attributes(self):
        if self.input_indices is None:
            raise ValueError("input_indices must be defined by an ablation dataset.")

        self.n_inputs = len(self.input_indices)
        self.n_outputs = 6
        self.j7 = []
        self.imu_idx = None
        self.jtorque_idx = [
            new_idx
            for new_idx, original_idx in enumerate(self.input_indices)
            if 18 <= original_idx < 24
        ]
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = self.wojtorque.clone()

    def _load_episode_from_datafiles(self, path):
        data_input, data_output, channel_mask = super()._load_episode_from_datafiles(
            path
        )
        data_input = data_input[:, self.input_indices]
        return data_input, data_output, channel_mask


# Without qddot
class HydraulicDatasetRelPos6DNoJointAcc(_HydraulicDatasetRelPos6DInputAblation):
    input_indices = tuple(range(0, 12)) + tuple(range(18, 30))


# Without qdot
class HydraulicDatasetRelPos6DNoJointVel(_HydraulicDatasetRelPos6DInputAblation):
    input_indices = tuple(range(0, 6)) + tuple(range(12, 30))


# Without qdot and qddot
class HydraulicDatasetRelPos6DNoJointVelAcc(
    _HydraulicDatasetRelPos6DInputAblation
):
    input_indices = tuple(range(0, 6)) + tuple(range(18, 30))


# Without u
class HydraulicDatasetRelPos6DNoJTorque(_HydraulicDatasetRelPos6DInputAblation):
    input_indices = tuple(range(0, 18)) + tuple(range(24, 30))


# Positions only (dq and q0)
class HydraulicDatasetRelPos6DJointPosOnly(_HydraulicDatasetRelPos6DInputAblation):
    input_indices = tuple(range(0, 6)) + tuple(range(24, 30))

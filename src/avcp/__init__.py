"""avcp — active vibration control of a thin-walled plate during milling."""
from .params import (SystemParams, PlateParams, PiezoParams, SensorParams,
                     MillingParams, ControlParams, ModelConfig)
from .fem import PlateFem
from .modal import ModalModel, build_modal_model, removal_thickness_map
from .milling import MillingForce
from .simulate import simulate, SimResult
from .controllers import (PDController, PshlqgController, PshlqgConfig,
                          tune_vpd, tune_cpd, tune_r_u, quintic_fit,
                          pd_loop_frf)

__all__ = [
    "SystemParams", "PlateParams", "PiezoParams", "SensorParams",
    "MillingParams", "ControlParams", "ModelConfig", "PlateFem",
    "ModalModel", "build_modal_model", "removal_thickness_map",
    "MillingForce", "simulate", "SimResult", "PDController",
    "PshlqgController", "PshlqgConfig", "tune_vpd", "quintic_fit",
    "pd_loop_frf",
]

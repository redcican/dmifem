"""Dynamic multi-sensor inverse finite element method (dmifem).

The iFEM strain functional is the observation model of a kinematic state-space estimator;
see the module docstrings for the correspondence with the manuscript sections.
"""

from dmifem.elements.beam import InverseBeam
from dmifem.estimator import DynamicMultiSensorIFEM, Hyperparameters
from dmifem.mesh import Mesh
from dmifem.sensors import (
    Accelerometer,
    DisplacementSensor,
    Inclinometer,
    SensorData,
    SensorType,
    StrainGauge,
)

__all__ = [
    "Accelerometer",
    "DisplacementSensor",
    "DynamicMultiSensorIFEM",
    "Hyperparameters",
    "Inclinometer",
    "InverseBeam",
    "Mesh",
    "SensorData",
    "SensorType",
    "StrainGauge",
]

__version__ = "0.1.0"

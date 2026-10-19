"""Dynamic multi-sensor inverse finite element method (dmifem).

The iFEM strain functional is the observation model of a kinematic state-space estimator in
dimensionless coordinates, with a correlated prior on the stress resultants, a K1-orthonormal
reduced basis that makes the reduction exact, and the exact marginal likelihood; see the module
docstrings for the correspondence with the manuscript sections.
"""

from dmifem.elements.beam import InverseBeam
from dmifem.estimator import DynamicMultiSensorIFEM, Hyperparameters
from dmifem.mesh import Mesh
from dmifem.prior import BeamSection
from dmifem.scaling import Scaling
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
    "BeamSection",
    "DisplacementSensor",
    "DynamicMultiSensorIFEM",
    "Hyperparameters",
    "Inclinometer",
    "InverseBeam",
    "Mesh",
    "Scaling",
    "SensorData",
    "SensorType",
    "StrainGauge",
]

__version__ = "0.2.0"

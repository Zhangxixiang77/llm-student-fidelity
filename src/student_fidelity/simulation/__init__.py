"""Student-simulation interfaces and execution utilities."""

from student_fidelity.simulation.huggingface import HuggingFaceLocalSimulator
from student_fidelity.simulation.personas import Persona, attribute_personas
from student_fidelity.simulation.runner import run_simulation
from student_fidelity.simulation.simulators import (
    DINAMockSimulator,
    Item,
    OpenAICompatibleSimulator,
    SimulationAnswer,
    StudentSimulator,
)

__all__ = [
    "DINAMockSimulator",
    "HuggingFaceLocalSimulator",
    "Item",
    "OpenAICompatibleSimulator",
    "Persona",
    "SimulationAnswer",
    "StudentSimulator",
    "attribute_personas",
    "run_simulation",
]

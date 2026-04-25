"""
NyayaRL - Indian Legal Reasoning Arena

Multi-agent RL environment for training AI to perform legal reasoning
in Indian criminal courts.
"""

from nyayarl.precedents import PrecedentsDB
from nyayarl.agents import DefenceAgent, JudgeAgent, ProsecutionAgent

__version__ = "0.1.0"
__all__ = ["PrecedentsDB", "JudgeAgent", "ProsecutionAgent", "DefenceAgent"]

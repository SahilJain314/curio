"""
The following code is taken from:
https://www.notinventedhere.org/articles/python/how-to-use-strings-as-name-aliases-in-python-enums.html
"""
from enum import Enum
import itertools

_CHANNELS = {
    0: ["UCEBb1b_L6zDS3xTUrIALZOw", "MIT"],
    1: ["3 Blue 1 Brown", "TBOB"],
    2: ["Blackpenredpen", "BPRP"],
    3: ["Crash Course", "CRASH"],
    4: ["The Teachlead", "TECHLEAD"],
    5: ["HackerRank", "HACKRANK"]
}

Channel = Enum(
    value="Channel",
    names=itertools.chain.from_iterable(
        itertools.product(v, [k]) for k, v in _CHANNELS.items()
    )
)
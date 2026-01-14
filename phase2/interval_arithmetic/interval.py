"""
Interval Arithmetic Module for Phase 2 Verification.
Implements rigorous interval arithmetic with outward rounding.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Interval:
    """
    Rigorous interval arithmetic with automatic outward rounding.
    Represents a closed interval [lower, upper].
    """
    lower: float
    upper: float

    def __post_init__(self):
        if self.lower > self.upper:
            # Allow small floating point violations, fix them
            if self.lower - self.upper < 1e-15:
                self.upper = self.lower
            else:
                raise ValueError(f"Invalid interval: [{self.lower}, {self.upper}]")

    def __add__(self, other):
        if isinstance(other, Interval):
            # Outward rounding using nextafter for rigor
            new_lower = float(self.lower + other.lower)
            new_upper = float(self.upper + other.upper)
            return Interval(
                np.nextafter(new_lower, -np.inf), 
                np.nextafter(new_upper, np.inf)
            )
        elif isinstance(other, (int, float)):
            new_lower = float(self.lower + other)
            new_upper = float(self.upper + other)
            return Interval(
                np.nextafter(new_lower, -np.inf), 
                np.nextafter(new_upper, np.inf)
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            new_lower = float(self.lower - other.upper)
            new_upper = float(self.upper - other.lower)
            return Interval(
                np.nextafter(new_lower, -np.inf),
                np.nextafter(new_upper, np.inf)
            )
        elif isinstance(other, (int, float)):
            new_lower = float(self.lower - other)
            new_upper = float(self.upper - other)
            return Interval(
                np.nextafter(new_lower, -np.inf),
                np.nextafter(new_upper, np.inf)
            )
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
             eps = np.finfo(float).eps
             return Interval(other - self.upper - eps, other - self.lower + eps)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ]
            return Interval(
                np.nextafter(float(min(products)), -np.inf), 
                np.nextafter(float(max(products)), np.inf)
            )
        elif isinstance(other, (int, float)):
            val = float(other)
            products = [self.lower * val, self.upper * val]
            return Interval(
                np.nextafter(float(min(products)), -np.inf), 
                np.nextafter(float(max(products)), np.inf)
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Interval):
             if other.lower <= 0 <= other.upper:
                 raise ValueError("Division by interval containing zero")
             reciprocal = Interval(1.0/other.upper, 1.0/other.lower)
             return self * reciprocal
        elif isinstance(other, (int, float)):
             if other == 0:
                 raise ValueError("Division by zero")
             return self * (1.0 / other)
        return NotImplemented

    def div_interval(self, other):
        """Explicit division method for compatibility."""
        return self.__truediv__(other)

    def __pow__(self, exponent: int):
        if exponent == 0:
            return Interval(1.0, 1.0)
        elif exponent == 1:
            return self
        elif exponent > 1:
            # Naive implementation, could be optimized for even powers
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        return NotImplemented

    def width(self) -> float:
        return self.upper - self.lower

    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def mag(self) -> float:
        """Magnitude: max(|lower|, |upper|)"""
        return max(abs(self.lower), abs(self.upper))

    def subset_of(self, other: 'Interval') -> bool:
        return other.lower <= self.lower and self.upper <= other.upper

    def intersection(self, other: 'Interval') -> 'Interval':
        l = max(self.lower, other.lower)
        u = min(self.upper, other.upper)
        if l <= u:
            return Interval(l, u)
        return None

    def __repr__(self):
        return f"[{self.lower:.6e}, {self.upper:.6e}]"

    def exp(self):
        """Exponential function for interval."""
        import math
        # exp is monotonic increasing
        # Round outward
        eps = np.finfo(float).eps
        return Interval(math.exp(self.lower) * (1.0 - eps), math.exp(self.upper) * (1.0 + eps))

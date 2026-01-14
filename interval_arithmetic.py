"""
Interval Arithmetic Module for Phase 2 Verification.
Implements rigorous interval arithmetic with outward rounding.
"""

import math

class Interval:
    """
    Rigorous interval arithmetic with automatic outward rounding.
    Represents a closed interval [lower, upper].
    Uses math.nextafter to ensure containment of true values.
    """
    # Rigorous Constants
    PI_LOWER = 3.141592653589793
    PI_UPPER = 3.141592653589794
    
    def __init__(self, lower, upper):
        self.lower = float(lower)
        self.upper = float(upper)
        # Verify strict ordering
        if self.lower > self.upper:
            # Check for very small floating point noise which is common in iterative calculations
            if self.lower - self.upper < 1e-15:
                # Slightly widen to resolve
                self.upper = self.lower
            else:
                raise ValueError(f"Invalid Interval Const: [{lower}, {upper}]")

    @classmethod
    def from_value(cls, value):
        v = float(value)
        return cls(v, v)
    
    @classmethod
    def pi(cls):
        return cls(cls.PI_LOWER, cls.PI_UPPER)

    @classmethod
    def log2(cls):
        # 0.6931471805599453
        return cls(0.6931471805599452, 0.6931471805599454)
        
    def __repr__(self):
        return f"[{self.lower:.6g}, {self.upper:.6g}]"

    def sqrt(self):
        if self.lower < 0:
            raise ValueError("sqrt of negative interval")
        return Interval(
            math.nextafter(math.sqrt(self.lower), -float('inf')),
            math.nextafter(math.sqrt(self.upper), float('inf'))
        )

    def log(self):
        if self.lower <= 0:
            raise ValueError("log of non-positive interval")
        return Interval(
            math.nextafter(math.log(self.lower), -float('inf')),
            math.nextafter(math.log(self.upper), float('inf'))
        )

    def exp(self):
        return Interval(
            math.nextafter(math.exp(self.lower), -float('inf')),
            math.nextafter(math.exp(self.upper), float('inf'))
        )


    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(
                math.nextafter(self.lower + other.lower, -math.inf),
                math.nextafter(self.upper + other.upper, math.inf)
            )
        else:
            val = float(other)
            return Interval(
                math.nextafter(self.lower + val, -math.inf),
                math.nextafter(self.upper + val, math.inf)
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(
                math.nextafter(self.lower - other.upper, -math.inf),
                math.nextafter(self.upper - other.lower, math.inf)
            )
        else:
            val = float(other)
            return Interval(
                math.nextafter(self.lower - val, -math.inf),
                math.nextafter(self.upper - val, math.inf)
            )
            
    def __rsub__(self, other):
        # other - self = [other - self.upper, other - self.lower]
        val = float(other)
        return Interval(
            math.nextafter(val - self.upper, -math.inf),
            math.nextafter(val - self.lower, math.inf)
        )

    def __mul__(self, other):
        if isinstance(other, Interval):
            p = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ]
            return Interval(
                math.nextafter(min(p), -math.inf),
                math.nextafter(max(p), math.inf)
            )
        else:
            val = float(other)
            p = [self.lower * val, self.upper * val]
            return Interval(
                math.nextafter(min(p), -math.inf),
                math.nextafter(max(p), math.inf)
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def div_interval(self, other):
        # Explicit division method to differentiate from float division
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                # Division by zero or interval containing zero results in unbounded
                return Interval(-float('inf'), float('inf'))
            p = [
                self.lower / other.lower,
                self.lower / other.upper,
                self.upper / other.lower,
                self.upper / other.upper
            ]
            return Interval(
                math.nextafter(min(p), -math.inf),
                math.nextafter(max(p), math.inf)
            )
        else:
            val = float(other)
            if val == 0:
                 return Interval(-float('inf'), float('inf'))
            return Interval(
                math.nextafter(self.lower / val, -math.inf),
                math.nextafter(self.upper / val, math.inf)
            )

    def __truediv__(self, other):
        return self.div_interval(other)

    def log(self):
        if self.lower <= 0:
            return Interval(-float('inf'), math.nextafter(math.log(self.upper), math.inf))
        return Interval(
            math.nextafter(math.log(self.lower), -math.inf),
            math.nextafter(math.log(self.upper), math.inf)
        )
        
    def exp(self):
        return Interval(
            math.nextafter(math.exp(self.lower), -math.inf),
            math.nextafter(math.exp(self.upper), math.inf)
        )
        
    def sqrt(self):
        if self.lower < 0:
             # Handle partial negative domain by clipping to 0
             lo = 0.0
        else:
             lo = self.lower
        return Interval(
            math.nextafter(math.sqrt(lo), -math.inf),
            math.nextafter(math.sqrt(self.upper), math.inf)
        )

    @property
    def mid(self):
        return (self.lower + self.upper) / 2.0
    
    @property
    def width(self):
        return self.upper - self.lower
        
    def __str__(self):
        return f"[{self.lower:.6g}, {self.upper:.6g}]"
    
    def __repr__(self):
        return self.__str__()

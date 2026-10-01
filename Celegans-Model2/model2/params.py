"""Model constants. Units: mm, s, oxygen as a fraction (0.21 = 21 %).

Paper values are from Demir et al., eLife 2020, Materials and methods (parameter list
below Eq. 47), converted from SI. Assumption numbers (A1...) refer to DESIGN_NOTES.md.
"""
from dataclasses import dataclass

# Domain and numerics
BOX = 20.0            # mm, side of the periodic box (paper: L = 2 cm)
GRID = 512            # cells per side for oxygen, dx = 0.039 mm
DENSITY_GRID = 128    # cells per side for worm density, 0.156 mm (smoothed anyway, A6)
DT = 0.05             # s, physics time step (fixed)
SMOOTHING = 0.25      # mm, sigma of the worm-density footprint (A6, numerical)

# Paper parameters
TAU = 0.5             # 1/s, tumbling rate
D_O = 2e-3            # mm^2/s, oxygen diffusion (2e-5 cm^2/s)
F = 0.65              # 1/s, oxygen penetration from the air
K_C = 7.3e-4          # 1/s per (worm/mm^2); 7.3e-10 /s with W in worms/m^2 (A2)

AMBIENT_STEPS = (0.01, 0.03, 0.07, 0.10, 0.14, 0.17, 0.21)   # Fig. 2b points (A10)
DENSITY_MAX = 90      # worms/mm^2, top of the paper's range


@dataclass(frozen=True)
class Strain:
    """V(O) = a O^2 + b O + c (paper Eq. 47), coefficients in m/s."""
    name: str
    a: float
    b: float
    c: float
    source: str

    def speed(self, o):
        """Speed in mm/s at oxygen fraction o."""
        return (self.a * o * o + self.b * o + self.c) * 1e3


NPR1 = Strain('npr-1', 1.89e-2, -3.98e-3, 2.25e-4, 'paper Eq. 47')          # A1, A3
N2 = Strain('N2', 9.02e-3, -2.64e-3, 2.07e-4, 'fitted to Fig. 2b')         # A4
STRAINS = (NPR1, N2)



def unstable(strain, density, ambient, f=F, k_c=K_C):
    """Paper's criterion for the uniform state: W beta k_c > f D_W."""
    o = ambient - k_c * density / f
    if o < 0:
        return False
    v = strain.speed(o)
    dv = (2 * strain.a * o + strain.b) * 1e3
    return density * (v * dv / (2 * TAU)) * k_c > f * v * v / (2 * TAU)

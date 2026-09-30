# frameA and frameB, the two frames an object on SO(3) maps between, as the Julia port names them:
# frameA the one it maps from, the crystal, frameB the one it maps to, the specimen or a second
# crystal. An orientation carries them itself; a function, a field, a fibre or a grid on SO(3)
# answers to them over its CS and SS, which stay as the shorter names
from .install import install

FRAME_A = property(lambda self: self.CS, lambda self, fr: setattr(self, "CS", fr),
                   doc="the frame the orientations map from, the crystal frame; CS is its short name (MATLAB's SRight)")
FRAME_B = property(lambda self: self.SS, lambda self, fr: setattr(self, "SS", fr),
                   doc="the frame the orientations map to, the specimen or a second crystal frame; SS is its short name "
                       "(MATLAB's SLeft)")


def installFramePair(classes):
  """frameA and frameB attached to classes holding CS and SS"""
  install(classes, FRAME_A, "frameA")
  install(classes, FRAME_B, "frameB")


__all__ = []

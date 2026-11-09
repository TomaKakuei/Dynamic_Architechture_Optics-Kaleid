"""Original calibrated camera applied only to the neural estimate."""
from camera import Camera,WaveFit
from common import HERE,np
import json

class ForwardCamera:
    def __init__(self):
        self.camera = Camera(calibration=json.loads((HERE/'camera_calibration.json').read_text()))
        self.fit = WaveFit()

    def expected_or_fallback(self, estimate):
        try:
            rows = self.fit.fit(estimate)
            intensity,grids = [],[]
            for row in rows:
                slope = self.camera.phase_slope(row)
                if slope > 1800:raise ValueError('Original camera bandwidth exceeded')
                n = 2048 if slope <= 450 else (4096 if slope <= 900 else 8192)
                grids.append(n)
                intensity.append(self.camera.integrated_irradiance(row,2048) if n == 2048 else
                    self.camera.reference_irradiance(row,n,carrier_shift=True))
            expected = np.maximum(np.asarray(intensity)*self.camera.gain_e*self.camera.prnu,0)
            adu = (4*np.clip(expected+self.camera.dsnu,0,10000)+100).astype(np.float32)
            return adu,True,dict(grids=grids)
        except (ValueError,RuntimeError) as exc:
            if isinstance(exc,RuntimeError) and not str(exc).startswith('Continuous phase fit failed'):
                raise
            # A model-domain failure is retained and receives the original prior.
            # No truth-dependent fallback and no discarded training/test case.
            return np.zeros((9,128,128),np.float32),False,dict(error=repr(exc))

"""Fixed, dimensionless training losses; original fixed plane and PT removal."""
import torch

ARMS = {
    'direct_wrms': dict(w=1., p=0., q=0., d=0.),
    'psf_l2': dict(w=0., p=1., q=0., d=0.),
    'q2000_gap': dict(w=0., p=0., q=1., d=0.),
    'psf_prediction': dict(w=0., p=1., q=0., d=1.),
    'psf_q2000': dict(w=0., p=1., q=1., d=0.),
    'psf_prediction_q2000': dict(w=0., p=1., q=1., d=1.),
}


class OpticalLoss:
    def __init__(self, optic, arm, out=None):
        self.optic, self.weights = optic, ARMS[arm]
        self.image = None
        if self.weights['d']:
            from image_consistency import ImageConsistency
            self.image = ImageConsistency(out)
        self.reference_psf = self.reference_q = None
        if self.weights['p'] or self.weights['q']:
            with torch.no_grad():
                phase, mask = optic.residual(torch.zeros((1, 40), device='cuda', dtype=torch.float64))
                self.reference_psf, self.reference_q = self.diffraction(phase, mask)
                self.reference_energy = self.reference_psf.square().sum((-1, -2)).clamp_min(1e-30)

    def diffraction(self, phase, mask):
        op = self.optic
        pupil = torch.zeros((*phase.shape[:2], op.grid, op.grid), device=phase.device, dtype=torch.complex128)
        pupil[:, :, op.disk] = torch.exp(2j*torch.pi*phase)*mask
        power = torch.fft.fft2(pupil, s=(2*op.grid, 2*op.grid)).abs().square()
        psf = power / power.sum((-1, -2), keepdim=True).clamp_min(1e-30)
        # Same physical frequency interpolation and field aggregation as Optics.metrics.
        otf = torch.fft.ifft2(power)
        n = int(op.freq_index); f = op.freq_index-n
        mtf = ((1-f)*otf[:, :, n, 0]+f*otf[:, :, n+1, 0]).abs()/otf[:, :, 0, 0].real.clamp_min(1e-30)
        mean = mtf.mean(-1)
        q = mean-.3*((mtf-mean[:, None]).square().mean(-1)+1e-20).sqrt()
        return psf, q

    def __call__(self, phase, mask, wrms, command, initial, observed=None, ids=None, epoch=None):
        w = self.weights
        terms = {'w': (wrms/.07).mean(-1)}
        if w['p'] or w['q']:
            psf, q = self.diffraction(phase, mask)
            terms['p'] = ((psf-self.reference_psf).square().sum((-1, -2))/self.reference_energy).mean(-1)
            terms['q'] = ((q-self.reference_q)/self.reference_q).square()
        if self.image is not None:
            assert observed is not None and ids is not None and epoch is not None
            terms['d'] = self.image(command, observed, ids, epoch)
        loss = sum(w[k]*value for k, value in terms.items() if w[k])
        return loss, {k: float(v.detach().mean()) for k, v in terms.items()}

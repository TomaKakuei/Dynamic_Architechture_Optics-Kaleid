"""Frozen numerical camera integrator from Mod3; source SHA256 dcc45c61012298ebfa76532609856a9f9a561f92966dceb49646661b43a9dc48.
Noise and exposure are replaced in camera.py.
"""
import numpy as np
import torch
import scipy.ndimage as ndi
from scipy.interpolate import CubicSpline
class Integrator:
    def resources(self,n):
        if n in self.cache:return self.cache[n]
        # Ray samples include endpoints; field integration uses pixel centers.
        coord=(np.arange(n)+.5)*2/n-1
        v=np.polynomial.chebyshev.chebvander(coord,self.degree)
        xx,yy=np.meshgrid(coord,coord)
        radii=np.hypot(xx,yy);angle=np.mod(np.arctan2(yy,xx),2*np.pi)
        big=2*n
        # Fourier interpolation preserves the physical diffuser footprint.
        from scipy.signal import resample
        centered=np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(self.engine.diffuser),
                              shift=(-32./n,-32./n))).real
        mask=resample(resample(centered,big,axis=0),big,axis=1).real
        dx=self.camera_diameter_mm/n
        freq=np.fft.fftfreq(big,dx);fx,fy=np.meshgrid(freq,freq)
        wavelength=self.engine.wave*1e-6
        term=1-(wavelength*fx)**2-(wavelength*fy)**2
        # Complex root attenuates evanescent components instead of aliasing them.
        transfer=np.exp(2j*np.pi/wavelength*self.propagation_mm*(np.sqrt(term.astype(complex))-1))
        row=dict(v=torch.tensor(v,dtype=torch.float64,device=self.device),radii=radii,angle=angle,
                 coords=coord,
                 fx=torch.tensor(fx,dtype=torch.float32,device=self.device),
                 fy=torch.tensor(fy,dtype=torch.float32,device=self.device),
                 diffuser=torch.tensor(np.exp(1j*mask),dtype=torch.complex64,device=self.device),
                 transfer=torch.tensor(transfer,dtype=torch.complex64,device=self.device))
        self.cache[n]=row
        return row

    def coverage_array(self,resource,boundary,n):
        angles=np.arange(len(boundary)+1)*2*np.pi/len(boundary)
        spline=CubicSpline(angles,np.r_[boundary,boundary[0]],bc_type='periodic')
        radial=np.minimum(spline(resource['angle']),1.)
        distance=radial-resource['radii'];coverage=(distance>0).astype(float)
        iy,ix=np.where(abs(distance)<np.sqrt(2)/n)
        x=resource['coords'][ix];y=resource['coords'][iy]
        counts=np.zeros(len(x))
        for sy in (np.arange(8)+.5)/8-.5:
            for sx in (np.arange(8)+.5)/8-.5:
                xx=x+sx*2/n;yy=y+sy*2/n
                rr=np.hypot(xx,yy);theta=np.mod(np.arctan2(yy,xx),2*np.pi)
                counts+=(rr<=np.minimum(spline(theta),1.))
        coverage[iy,ix]=counts/64.
        return coverage

    def aperture(self,resource,boundary,n):
        key=(n,boundary.tobytes())
        if key in self.aperture_cache:return self.aperture_cache[key]
        coverage=self.coverage_array(resource,boundary,n)
        value=torch.tensor(np.sqrt(coverage),dtype=torch.float64,device=self.device)
        if np.all(boundary==1.):self.aperture_cache[key]=value
        return value

    def integrate_field(self,representation,n,carrier_shift=True):
        resource=self.resources(n);v=resource['v']
        coef=torch.tensor(representation['coefficients'],dtype=torch.float64,device=self.device)
        cx,cy=representation['carrier']
        if carrier_shift:coef[1,0]-=cx;coef[0,1]-=cy
        phase=v@coef.T@v.T
        amplitude=self.aperture(resource,representation['boundary'],n)
        pupil=torch.zeros((2*n,2*n),dtype=torch.complex64,device=self.device)
        angle=torch.remainder(2*np.pi*phase,2*np.pi).to(torch.float32)
        pupil[n//2:3*n//2,n//2:3*n//2]=torch.polar(amplitude.to(torch.float32),angle)
        transfer=resource['transfer']
        if carrier_shift:
            # Modulation theorem: preserve the exact carrier in H(f+f0).
            # The omitted output carrier has unit modulus, so intensity is
            # unchanged. No physical tilt is removed from the camera image.
            wavelength=self.engine.wave*1e-6
            fx=resource['fx']+2*cx/self.camera_diameter_mm
            fy=resource['fy']+2*cy/self.camera_diameter_mm
            r2=(wavelength*fx)**2+(wavelength*fy)**2
            root=torch.sqrt(torch.clamp(1-r2,min=0.))
            kz=2*np.pi/wavelength*self.propagation_mm
            # Rationalized sqrt(1-r2)-1 avoids subtractive cancellation.
            phase_h=torch.where(r2<=1.,-kz*r2/(root+1.),-kz)
            decay=torch.exp(-kz*torch.sqrt(torch.clamp(r2-1,min=0.)))
            transfer=torch.polar(decay,phase_h)
        field=torch.fft.ifft2(torch.fft.fft2(pupil*resource['diffuser'])*transfer)
        factor=2*n//self.sensor_size
        intensity=field.abs().square().reshape(self.sensor_size,factor,self.sensor_size,factor).mean(dim=(1,3))
        return intensity.cpu().numpy()

    def integrated_irradiance(self,representation,n=2048):
        return self.integrate_field(representation,n)

    def reference_irradiance(self,representation,n=4096,carrier_shift=False):
        """Bounded-memory CPU reference, with optional full physical carrier.

        Independent implementation used only to validate the production
        renderer. Chunk the transfer function instead of allocating several
        8192-square double-complex work arrays on the small laptop GPU.
        """
        from scipy import fft as sf
        from scipy.signal import resample
        coord=(np.arange(n)+.5)*2/n-1
        v=np.polynomial.chebyshev.chebvander(coord,self.degree)
        xx,yy=np.meshgrid(coord,coord)
        geometry=dict(coords=coord,radii=np.hypot(xx,yy),angle=np.mod(np.arctan2(yy,xx),2*np.pi))
        del xx,yy
        coverage=self.coverage_array(geometry,representation['boundary'],n)
        del geometry
        coef=representation['coefficients'].copy();cx,cy=representation['carrier']
        if carrier_shift:coef[1,0]-=cx;coef[0,1]-=cy
        phase=v@coef.T@v.T
        big=2*n;field=np.zeros((big,big),np.complex64)
        field[n//2:3*n//2,n//2:3*n//2]=np.sqrt(coverage)*np.exp(2j*np.pi*phase)
        del coverage,phase
        centered=np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(self.engine.diffuser),
                              shift=(-32./n,-32./n))).real.astype(np.float32)
        mask=resample(resample(centered,big,axis=0),big,axis=1).real
        field*=np.exp(np.complex64(1j)*mask)
        del mask
        spectrum=sf.fft2(field,overwrite_x=True,workers=4)
        del field
        wavelength=self.engine.wave*1e-6;freq=np.fft.fftfreq(big,self.camera_diameter_mm/n)
        offsetx=2*cx/self.camera_diameter_mm if carrier_shift else 0.
        offsety=2*cy/self.camera_diameter_mm if carrier_shift else 0.
        for start in range(0,big,32):
            fx=freq[None,:]+offsetx;fy=freq[start:start+32,None]+offsety
            root=np.sqrt((1-(wavelength*fx)**2-(wavelength*fy)**2).astype(complex))
            h=np.exp(2j*np.pi/wavelength*self.propagation_mm*(root-1)).astype(np.complex64)
            spectrum[start:start+32]*=h
        field=sf.ifft2(spectrum,overwrite_x=True,workers=4)
        factor=big//self.sensor_size
        intensity=(field.real**2+field.imag**2).reshape(128,factor,128,factor).mean(axis=(1,3))
        return intensity

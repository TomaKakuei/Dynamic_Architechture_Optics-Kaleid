"""Known-camera image response at estimated poses, never hidden wavefront input.

Original relay/diffuser/ADU model;1024pupil/2048FFT numerical rendering.
Ray-phase response uses35 central differences; propagation response is analytic.
Pupil boundary is refreshed at each proposed pose, frozen in the local Jacobian.
"""
from support import *


class CameraModel:
    def __init__(self,n=1024,degree=24):
        self.camera=RelayCamera(device='cuda',degree=degree)
        self.optics=TensorOptics(self.camera.engine,pupil_grid=64,device='cuda')
        pupils=np.vstack(([0.,0.],self.camera.xy))
        launches=[self.camera.engine.launch(field,pupils) for field in self.camera.engine.fields]
        self.optics.launch_points=torch.as_tensor(np.stack([x[0] for x in launches]),device='cuda').double()
        self.optics.launch_directions=torch.as_tensor(np.stack([x[1] for x in launches]),device='cuda').double()
        self.n=n;self.degree=degree;self.resource=self.camera.resources(n)
        self.pinvs={};self.terms=self.camera.terms
        self.last_diagnostics={}

    def phase(self,q):
        o=self.optics;p,d,opl,valid=o.trace(q.double())
        chief=valid[:,:,0];p,d,opl,valid=p[:,:,1:],d[:,:,1:],opl[:,:,1:],valid[:,:,1:]
        offset=p-o.reference_centers[None,:,None,:]
        radius2=((o.reference_centers-o.reference_exits)**2).sum(-1)[None,:,None]
        beta=(offset*d).sum(-1);cc=offset.square().sum(-1)-radius2
        disc=beta.square()-cc;valid=valid&(disc>0)
        root=disc.clamp_min(1e-30).sqrt();den=-beta+root
        small=torch.where(den.abs()>1e-12,cc/torch.where(den.abs()>1e-12,den,torch.ones_like(den)),-beta-root)
        large=-beta+root;distance=torch.where(small.abs()<large.abs(),small,large)
        # Keep finite continuation outside apertures for a fixed-mask local derivative.
        phase=(opl+distance-o.reference_opls[None,:,None])*o.n_image/o.wavelength_mm
        return phase,valid,chief

    @torch.no_grad()
    def representations(self,q,derivative):
        q=np.asarray(q,dtype=np.float64);h=1e-4
        poses=np.r_[q[None],q[None]+h*np.eye(35),q[None]-h*np.eye(35)] if derivative else q[None]
        chunks=[];central_valid=None
        for first in range(0,len(poses),8):
            p,v,chief=self.phase(torch.as_tensor(poses[first:first+8],device='cuda'))
            chunks.append(p)
            if first==0:
                central_valid=v[0].cpu().numpy()
                if not bool(chief[0].all()):raise ValueError('Estimated pose has invalid chief ray')
        phase=torch.cat(chunks);rows=[];fit=[]
        for fi in range(9):
            valid=central_valid[fi]
            if valid[:self.camera.base_count].mean()<.2:raise ValueError('Estimated pupil support below20percent')
            key=(fi,valid.tobytes())
            if key not in self.pinvs:
                design=self.camera.design[valid]
                pinv=np.linalg.pinv(design,rcond=1e-12)
                linear=np.linalg.pinv(np.c_[np.ones(valid.sum()),self.camera.xy[valid]],rcond=1e-12)
                if len(self.pinvs)>36:self.pinvs.clear()
                self.pinvs[key]=(torch.as_tensor(pinv,device='cuda'),torch.as_tensor(linear,device='cuda'))
            pinv,linear=self.pinvs[key]
            values=phase[:,fi,valid];values=values-values.mean(1,keepdim=True)
            coef=values@pinv.T;carrier=(values@linear.T)[:,1:]
            err=values[0]-torch.as_tensor(self.camera.design[valid],device='cuda')@coef[0]
            fit.append(float(err.square().mean().sqrt()))
            matrix=torch.zeros((len(poses),self.degree+1,self.degree+1),device='cuda',dtype=torch.float64)
            for k,(i,j) in enumerate(self.terms):matrix[:,i,j]=coef[:,k]
            boundary=self.camera.pupil_boundary(q,fi)
            row=dict(matrix=matrix[0],carrier=carrier[0],boundary=boundary)
            if derivative:
                row['dmatrix']=(matrix[1:36]-matrix[36:])/(2*h)
                row['dcarrier']=(carrier[1:36]-carrier[36:])/(2*h)
            rows.append(row)
        self.last_diagnostics=dict(phase_fit_rms=fit,minimum_support=float(central_valid[:,:self.camera.base_count].mean(1).min()),
                                   model_pupil_grid=self.n,derivative_boundary='Frozen locally; refreshed in every objective evaluation')
        return rows

    @torch.no_grad()
    def field(self,row,derivative):
        n=self.n;r=self.resource;v=r['v'];matrix=row['matrix'].clone()
        cx,cy=row['carrier'];matrix[1,0]-=cx;matrix[0,1]-=cy
        phase=v@matrix.T@v.T
        amplitude=self.camera.aperture(r,row['boundary'],n).float()
        pupil=torch.zeros((2*n,2*n),device='cuda',dtype=torch.complex64)
        inner=torch.polar(amplitude,torch.remainder(2*np.pi*phase,2*np.pi).float())
        pupil[n//2:3*n//2,n//2:3*n//2]=inner
        wave=self.camera.engine.wave*1e-6;kz=2*np.pi/wave*self.camera.propagation_mm
        fx=r['fx']+2*cx/self.camera.camera_diameter_mm;fy=r['fy']+2*cy/self.camera.camera_diameter_mm
        r2=(wave*fx).square()+(wave*fy).square();root=(1-r2).clamp_min(0).sqrt()
        phase_h=torch.where(r2<=1,-kz*r2/(root+1),torch.full_like(r2,-kz))
        decay=torch.exp(-kz*(r2-1).clamp_min(0).sqrt());transfer=torch.polar(decay.float(),phase_h.float())
        spectrum=torch.fft.fft2(pupil*r['diffuser']);field=torch.fft.ifft2(spectrum*transfer)
        factor=2*n//128
        def integrate(x):return x.reshape(*x.shape[:-2],128,factor,128,factor).mean(dim=(-3,-1))
        intensity=integrate(field.abs().square());expected=expected_adu(intensity)
        if not derivative:return expected,None
        index=int(intensity.flatten().argmax());peak=intensity.flatten()[index]
        columns=[]
        for first in range(0,35,3):
            dm=row['dmatrix'][first:first+3].clone();dc=row['dcarrier'][first:first+3]
            dm[:,1,0]-=dc[:,0];dm[:,0,1]-=dc[:,1]
            dp=v[None]@dm.transpose(1,2)@v.T[None]
            dpupil=torch.zeros((len(dm),2*n,2*n),device='cuda',dtype=torch.complex64)
            dpupil[:,n//2:3*n//2,n//2:3*n//2]=inner[None]*(2j*np.pi)*dp.float()
            dphase_h=-kz*wave**2/root.clamp_min(1e-6)*(fx[None]*dc[:,0,None,None]+fy[None]*dc[:,1,None,None])*2/self.camera.camera_diameter_mm
            dfield=torch.fft.ifft2(torch.fft.fft2(dpupil*r['diffuser'][None])*transfer[None]+
                spectrum[None]*transfer[None]*1j*dphase_h.float())
            di=integrate(2*(field.conj()[None]*dfield).real)
            dpeak=di.flatten(1)[:,index]
            derivative_adu=16000*(di/peak-intensity[None]*dpeak[:,None,None]/peak.square())
            columns.append(derivative_adu)
        return expected,torch.cat(columns).permute(1,2,0)

    @torch.no_grad()
    def evaluate(self,q,derivative=False):
        start=common.clock();rows=self.representations(q,derivative);ray=common.clock()-start
        images=[];jacs=[]
        for row in rows:
            image,jac=self.field(row,derivative);images.append(image)
            if derivative:jacs.append(jac)
        self.last_diagnostics.update(ray_seconds=ray,total_seconds=common.clock()-start)
        return torch.stack(images),torch.stack(jacs) if derivative else None

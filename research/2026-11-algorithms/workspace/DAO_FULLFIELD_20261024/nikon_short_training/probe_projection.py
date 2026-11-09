"""Brief interface check after a failed actual4epoch candidate."""
from short_train import TrialSystem,exp,HERE,np,torch
s=TrialSystem('native_projection');data=s.load_data(pilot=True);ix=np.arange(16)
m=exp.System.model(s,'large');m.load_state_dict(s.saved['model']);m.eval()
native=s.legacy.NikonTensorOptics(nodes=s.legacy.nikon35.NODES,
    fields=s.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cuda')
with torch.no_grad():
    endpoint,target,command,_=s.forward(m,data,ix);p=-command
    measured=torch.tensor(data['raw'][ix],device='cuda',dtype=torch.float64).flatten(1)/7.
    true_signal=native.metrics(target)['difference'].flatten(1)/7.
    pred_signal=native.metrics(p.double())['difference'].flatten(1)/7.
    linear_p=p@s.native_response.T
    dlinear=(measured.float()-linear_p)@s.native_inverse.T
    dnonlinear=(measured.float()-pred_signal.float())@s.native_inverse.T
    bounded=dnonlinear/torch.clamp(dnonlinear.abs().amax(1,keepdim=True)/.25,min=1.)
    rows={}
    for label,estimate in [('original',p),('linear',p+dlinear),('nonlinear_bounded',p+bounded)]:
        w,valid,_=s.metrics(target-estimate.clamp(-1.5,1.5).double(),target)
        rows[label]=dict(fullfield_pass=int(((w.amax(1)<.07)&valid).sum()),field_mean=w.mean(0).tolist())
    result=dict(count=len(ix),raw_vs_true_trace_rms_waves=float(((measured-true_signal)*7).square().mean().sqrt()),
        true_state_linearization_error_waves=float(((true_signal-target@s.native_response.double().T)*7).square().mean().sqrt()),
        predicted_nonlinear_measurement_error_waves=float(((measured-pred_signal)*7).square().mean().sqrt()),
        linear_delta_max=float(dlinear.abs().max()),nonlinear_delta_max=float(dnonlinear.abs().max()),results=rows)
exp.atomic(HERE/'projection_probe.json',result)
print(result)

"""One-packet nonlinear inverse layer with forward-mode AD response.

The forward optimizer uses the optical model only. Training uses an implicit
Gauss-Newton derivative at the terminal estimate (an approximation, not an
exact derivative of the four-step optimizer trajectory).
"""
import torch
import experiment as exp


class ExactResponseNetwork(exp.LargerNetwork):
    def difference(self,physical):
        # Native DIFF only: no reference-sphere calculation is needed here.
        optics=self.native_optics
        traced=optics.trace(physical.double()+optics.bias)
        return ((traced['opl'][:,:,1:]-optics.nominal_opl)@optics.projector.T/optics.wave).flatten(1)/7.

    def finite_response(self,physical):
        h=1e-3
        eye=torch.eye(35,dtype=torch.float64,device=physical.device)*h
        probes=torch.cat((physical[:,None]+eye,physical[:,None]-eye),dim=1)
        observed=self.difference(probes.reshape(-1,35)).reshape(len(physical),70,147)
        return ((observed[:,:35]-observed[:,35:])/(2*h)).transpose(1,2)

    @torch.no_grad()
    def response(self,physical):
        from fp32_response import response
        return response(self.native_optics,physical)

    @torch.no_grad()
    def captured_fd_response(self,physical):
        if not hasattr(self,'response_graphs'):self.response_graphs={}
        key=len(physical)
        if key not in self.response_graphs:
            static=physical.detach().clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.finite_response(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=self.finite_response(static)
            self.response_graphs[key]=(graph,static,output)
        graph,static,output=self.response_graphs[key]
        static.copy_(physical.detach());graph.replay()
        return output.clone()

    def forward(self,image,raw=None):
        prediction=super().forward(image,raw)
        prior=(prediction*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5).double()
        measured=raw.flatten(1).double()/7.
        identity=torch.eye(35,dtype=torch.float64,device=prior.device)
        ridge=1e-4
        with torch.no_grad():
            physical=prior.detach()
            for iteration in range(4):
                residual=self.difference(physical)-measured
                jac=self.response(physical);jt=jac.transpose(1,2)
                gram=jt@jac+ridge*identity
                rhs=(jt@residual.unsqueeze(-1)).squeeze(-1)+ridge*(physical-prior)
                delta=-torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
                delta=delta/torch.clamp(delta.abs().amax(1,keepdim=True)/.25,min=1.)
                objective=residual.square().sum(1)+ridge*(physical-prior).square().sum(1)
                origin=physical;accepted=torch.zeros(len(physical),dtype=torch.bool,device=physical.device)
                for gain in (1.,.5,.25):
                    rows=torch.nonzero(~accepted,as_tuple=True)[0]
                    if len(rows)==0:break
                    trial=(origin[rows]+gain*delta[rows]).clamp(-1.5,1.5)
                    trial_residual=self.difference(trial)-measured[rows]
                    merit=trial_residual.square().sum(1)+ridge*(trial-prior[rows]).square().sum(1)
                    good=merit<objective[rows]
                    if bool(good.any()):
                        selected=rows[good]
                        physical=physical.index_copy(0,selected,trial[good])
                        accepted[selected]=True
            if torch.is_grad_enabled():
                raise AssertionError('Forward optimizer must not retain ray-tracing graphs')
        if torch.is_grad_enabled():
            with torch.no_grad():
                terminal_jac=self.response(physical)
                terminal_gram=terminal_jac.transpose(1,2)@terminal_jac+ridge*identity
            # Implicit GN: dx/dprior ~= ridge*(J'J+ridge*I)^(-1).
            # Forward value is exactly the solved estimate; network parameters
            # remain FP32 and all remain trainable through the prior.
            implicit=torch.linalg.solve(terminal_gram,(ridge*(prior-prior.detach())).unsqueeze(-1)).squeeze(-1)
            physical=physical+implicit
        return ((physical-self.base.target_mean)/self.base.target_std).float()

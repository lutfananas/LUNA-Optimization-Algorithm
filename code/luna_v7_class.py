import sys, numpy as np, math
sys.path.insert(0, '/home/z/my-project/scripts')

class LUNA_v7:
    def __init__(self, **kw):
        defaults = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                 semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                 libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                 sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                 late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                 chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                 restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)
        defaults.update(kw)
        self.__dict__.update(defaults)
    def _ci(self, n, d, lb, ub):
        c = np.zeros((n, d))
        for i in range(n):
            x = 0.5
            for j in range(d): x = 4*x*(1-x); c[i,j] = lb + x*(ub-lb)
        return c
    def _rb(self, x, lb, ub):
        for _ in range(3): x = np.where(x>ub, 2*ub-x, x); x = np.where(x<lb, 2*lb-x, x)
        return np.clip(x, lb, ub)
    def optimize(self, f, dim, bounds, pop=30, max_iter=500, seed=None):
        if seed is not None: np.random.seed(seed)
        lb, ub = bounds
        G0=self.G0; Ns=self.N_syn; Na=self.N_anom; e=self.eccentricity
        a=self.semi_major; h=self.ang_momentum; la=self.libration_amp
        lf=self.libration_freq; sg=self.sun_gravity; od=self.orbital_decay
        si=self.sigma_init; sm=self.sigma_min; sx=self.sigma_max
        ldt=self.late_de_thresh; Fl=self.F_lo; Fh=self.F_hi
        cr=self.chaos_ratio; op=self.obl_period; of=self.obl_frac
        rp=self.restart_patience; rf=self.restart_frac; pw=self.pbest_weight; eps=self.eps
        nc = int(pop*cr); nr = pop-nc
        Xc = self._ci(nc, dim, lb, ub); Xr = np.random.uniform(lb, ub, (nr, dim))
        Xi = np.vstack([Xc, Xr]); Xo = (lb+ub)-Xi
        Xc2 = np.vstack([Xi, Xo]); fc = np.array([f(x) for x in Xc2])
        order = np.argsort(fc)[:pop]; X = Xc2[order].copy(); fit = fc[order].copy()
        pb = X.copy(); pbf = fit.copy()
        bi = np.argmin(fit); xb = X[bi].copy(); fb = fit[bi]; hh=[fb]
        sigma = si; sw=[]; li=0; lo=0
        for t in range(max_iter):
            ratio = t / max_iter; is_late = ratio > ldt
            G0_t = G0 * math.exp(-od * ratio)
            tp = 2*np.pi*Ns*t/max_iter; I = (1-np.cos(tp))/2
            ta = 2*np.pi*Na*t/max_iter; D = 1 + e*np.cos(ta)
            energy = max(2.0/D - 1.0/a, eps); v = np.sqrt(G0_t * energy); vf = v / (v + 1.0)
            Gt = G0_t / (D**2 + eps); kr = h / (D**2 + eps)
            ol = 2*np.pi*lf/max_iter; lib = la * np.sin(ol * t)
            tidal = Gt + sg * np.cos(tp); we = I; wx = 1 - I; ii = False
            for i in range(pop):
                diff = xb - X[i]; R = np.linalg.norm(diff) + eps; d = diff / R
                dp_v = pb[i] - X[i]; Rp = np.linalg.norm(dp_v) + eps; dp = dp_v / Rp
                if is_late:
                    F = np.random.uniform(Fl, Fh) * vf
                    cands = [j for j in range(pop) if j != i]; a2, b2 = np.random.choice(cands, 2, replace=False)
                    diff_ab = X[a2] - X[b2]
                    xn = X[i] + F * diff + F * Gt * diff_ab + pw * F * dp_v
                    if np.random.random() < 0.5:
                        D_abs = np.abs(xb - X[i]); l = np.random.uniform(-1, 1, dim); kr2 = kr * 2 * np.pi
                        xn = D_abs * np.exp(kr2 * l) * np.cos(kr2 * l) + xb
                elif I < 0.2:
                    rv = np.random.randn(dim); orth = rv - np.dot(rv, d) * d; orth /= (np.linalg.norm(orth) + eps)
                    xn = X[i] + lib * R * orth + sigma * np.random.randn(dim)
                elif I >= 0.8:
                    s = np.random.randint(1, 5)
                    if s == 1:
                        of2 = np.argsort(fit); al = X[of2[0]].copy()
                        be = X[of2[1]].copy() if len(of2)>1 else xb; de = X[of2[2]].copy() if len(of2)>2 else xb
                        Ra = np.linalg.norm(al-X[i])+eps; Rb = np.linalg.norm(be-X[i])+eps
                        Rd = np.linalg.norm(de-X[i])+eps; xi = np.random.uniform(0.7, 1.0)
                        pull = (0.5*v*(al-X[i])/Ra + 0.3*tidal*(be-X[i])/Rb + 0.2*(1+lib)*(de-X[i])/Rd + pw*v*dp_v/Rp)
                        xn = X[i] + xi * sigma * pull
                    elif s == 2:
                        D_abs = np.abs(xb - X[i]); l = np.random.uniform(-1, 1, dim); kr2 = kr * 2 * np.pi
                        xn = D_abs * np.exp(kr2 * l) * np.cos(kr2 * l) + xb
                    elif s == 3:
                        levy = np.random.randn(dim) / (np.abs(np.random.randn(dim)) + eps)**(1/1.5)
                        xn = xb - np.abs(2*np.random.random(dim)*xb - X[i]) * v * sigma * 0.1 + sigma * levy * 0.5
                    else:
                        cands = [j for j in range(pop) if j != i]; a2, b2 = np.random.choice(cands, 2, replace=False)
                        diff_ab = X[a2] - X[b2]; F = np.random.uniform(Fl, Fh) * vf
                        xn = X[i] + F * diff + F * Gt * diff_ab + pw * F * dp_v
                else:
                    cands = [j for j in range(pop) if j != i]; a2, b2 = np.random.choice(cands, 2, replace=False)
                    diff_ab = X[a2] - X[b2]; sa = sigma * vf
                    exploit = sa * diff + sigma * Gt * diff_ab
                    rv = np.random.randn(dim); orth = rv - np.dot(rv, d) * d; orth /= (np.linalg.norm(orth) + eps)
                    explore = lib * R * orth + sigma * np.random.randn(dim) * 0.5
                    xn = X[i] + we * exploit + wx * explore
                xn = self._rb(xn, lb, ub); fn = f(xn)
                if fn < fit[i]:
                    X[i]=xn; fit[i]=fn; ii=True
                    if fn < pbf[i]: pb[i]=xn.copy(); pbf[i]=fn
                    if fn < fb: fb=fn; xb=xn.copy(); li=t
            if not is_late:
                sw.append(1 if ii else 0)
                if len(sw)>20: sw.pop(0)
                if len(sw)==20:
                    sr = sum(sw)/20
                    if sr>0.2: sigma*=1.1
                    elif sr<0.2: sigma*=0.9
                    sigma = max(sm, min(sx, sigma))
                if t-lo >= op and t>0:
                    lo=t; n_obl=max(1,int(pop*of)); worst=np.argsort(fit)[-n_obl:]; opp=(lb+ub)-X[worst]
                    of3=np.array([f(x) for x in opp])
                    for k,idx in enumerate(worst):
                        if of3[k]<fit[idx]:
                            X[idx]=opp[k]; fit[idx]=of3[k]
                            if of3[k]<pbf[idx]: pb[idx]=opp[k].copy(); pbf[idx]=of3[k]
                            if of3[k]<fb: fb=of3[k]; xb=opp[k].copy(); li=t
                if t-li >= rp:
                    n_r=max(1,int(pop*rf)); worst=np.argsort(fit)[-n_r:]
                    for idx in worst:
                        X[idx]=np.random.uniform(lb,ub,dim); fit[idx]=f(X[idx])
                        pb[idx]=X[idx].copy(); pbf[idx]=fit[idx]
                        if fit[idx]<fb: fb=fit[idx]; xb=X[idx].copy(); li=t
            hh.append(fb)
        return xb, fb, hh

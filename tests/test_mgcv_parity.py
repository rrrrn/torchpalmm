"""Parity with mgcv::gam (REML) on simulated data; skipped without Rscript and mgcv.

Scaling: torchpalmm's smooth is f(x) = c0 + c1 x + B a with a ~ N(0, tau2 I), the mixed-model form of the
penalty  integral f''(x)^2 dx / tau2  (natural cubic spline, knots at every distinct x or at n_knots).
mgcv's bs="cr" with the same knots carries the same penalty, rescaled: it reports sp for S / S.scale, so
tau2 = sig2 * S.scale / sp, which is gam.vcomp()'s variance for the smooth. A bs="re" term gives D likewise.
"""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from torchpalmm import SmoothingSplineMixed

torch.set_default_dtype(torch.float64)
RSCRIPT = shutil.which("Rscript")
pytestmark = pytest.mark.skipif(
    RSCRIPT is None or subprocess.run([RSCRIPT, "-e", "library(mgcv)"], capture_output=True).returncode != 0,
    reason="needs R with mgcv")

R_CODE = """
library(mgcv)
d <- read.csv("{csv}"); d$id <- factor(d$id)
kn <- scan("{knots}", quiet = TRUE)
m <- gam(y ~ x + s(t, bs = "cr", k = length(kn)) + s(id, bs = "re"), data = d, method = "REML",
         knots = list(t = kn))
vc <- gam.vcomp(m)
tg <- sort(unique(d$t)); g <- d[rep(1, length(tg)), ]; g$t <- tg
f <- predict(m, g, type = "terms")[, "s(t)"]
out <- c(vc["scale", "std.dev"]^2, vc["s(t)", "std.dev"]^2, vc["s(id)", "std.dev"]^2, coef(m)[["x"]], f - mean(f))
writeLines(format(out, digits = 17), "{out}")
"""


@pytest.mark.parametrize("n_knots", [None, 12])
def test_matches_mgcv_gam(tmp_path, n_knots):
    import pandas as pd
    rng = np.random.default_rng(0)
    G, ni = 40, 8
    ids = np.repeat(np.arange(G), ni)
    t = np.round(rng.uniform(0, 1, G * ni), 2)
    x = rng.normal(size=G * ni)
    y = 0.5 * x + np.sin(2 * np.pi * t) + np.repeat(rng.normal(scale=0.7, size=G), ni) + rng.normal(scale=0.4, size=G * ni)

    ss = SmoothingSplineMixed(y=y, X=x[:, None], Z=np.ones((G * ni, 1)), spline_inputs=t, obs_id=ids,
                              n_knots=n_knots, device="cpu")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=1000, tol=1e-10, fit_type="reml")

    pd.DataFrame({"y": y, "x": x, "t": t, "id": ids}).to_csv(tmp_path / "d.csv", index=False)
    np.savetxt(tmp_path / "knots.txt", ss.knots[0].numpy(), fmt="%.17g")
    script = tmp_path / "fit.R"
    script.write_text(R_CODE.format(csv=(tmp_path / "d.csv").as_posix(), knots=(tmp_path / "knots.txt").as_posix(),
                                    out=(tmp_path / "out.txt").as_posix()))
    subprocess.run([RSCRIPT, str(script)], check=True, capture_output=True)
    r = np.loadtxt(tmp_path / "out.txt")
    sig2, tau2, var_id, beta_x, curve = r[0], r[1], r[2], r[3], r[4:]

    assert np.isclose(float(ss.sigma2), sig2, rtol=2e-2)
    assert np.isclose(float(ss.D[0, 0]), var_id, rtol=5e-2)
    assert np.isclose(float(ss.tau_sq_list[0]), tau2, rtol=0.1)
    assert np.isclose(float(ss.beta[0]), beta_x, atol=5e-3)
    ours = ss.smooth_confint()["theta_fitted"].to_numpy()          # centered over the distinct t
    assert np.abs(ours - curve).max() < 1e-2

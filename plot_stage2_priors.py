import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

Ly = 0.21

prior_means = np.array([1e8,  80.0, 35.0, 0.168, 0.021])   # theta_lab
start       = np.array([1e8, 120.0, 20.0, 0.189, 0.021])
truth       = np.array([2e8,  60.0, 40.0, 0.168, 0.021])

sigma_old = np.array([1.0, 0.30, 0.20, 0.50, 0.70])
sigma_new = np.array([1.0, 3.00, 4.00, 0.85, 0.70])

labels = [r'$\rho_\mathrm{melt1}$',
          r'$\kappa_\mathrm{surf}$',
          r'$\kappa_\mathrm{deep}$',
          r'$y_\mathrm{trans}$',
          r'$w$']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig, ax = plt.subplots(figsize=(9, 4.5))

log_devs = np.log(start / prior_means)
J_lik = 0.029

for i in range(5):
    s_old = sigma_old[i]
    s_new = sigma_new[i]
    xlim  = max(4*s_old, 4*s_new, 1.2)
    x     = np.linspace(-xlim, xlim, 600)

    pdf_old = norm.pdf(x, 0, s_old)
    pdf_new = norm.pdf(x, 0, s_new)

    ax.plot(x, pdf_old / pdf_old.max(), color=colors[i], lw=1.5, ls='--', alpha=0.6)
    ax.plot(x, pdf_new / pdf_new.max(), color=colors[i], lw=2.0, ls='-',
            label=f'{labels[i]}  σ: {s_old}→{s_new}')

    xs = log_devs[i]
    xt = np.log(truth[i] / prior_means[i])
    ax.axvline(xs, color=colors[i], lw=1.0, ls='--', alpha=0.5)
    ax.axvline(xt, color=colors[i], lw=1.2, ls=':',  alpha=0.8)

ax.axvline(np.nan, color='k', ls='--', lw=1.2, label='starting point')
ax.axvline(np.nan, color='k', ls=':',  lw=1.5, label='truth')
ax.axvline(0,      color='k', ls='-',  lw=0.8, alpha=0.25, label='prior mean')

# annotate prior/likelihood ratio
c_old = sum(0.5*(log_devs[i]/sigma_old[i])**2 for i in range(5))
c_new = sum(0.5*(log_devs[i]/sigma_new[i])**2 for i in range(5))
ax.set_title(
    f'Stage 2 log-normal priors\n'
    f'Prior/likelihood at start:  old σ → {c_old/J_lik:.0f}×  |  new σ → {c_new/J_lik:.1f}×',
    fontsize=10
)
ax.set_xlabel(r'$\log(\theta\,/\,\mu_\mathrm{prior})$  — log-deviation from prior mean')
ax.set_ylabel('Normalised prior density')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.set_xlim(-3, 3)
plt.tight_layout()
plt.savefig('stage2_priors.png', dpi=150)
print('saved stage2_priors.png')
print(f'Prior at start (old σ): {c_old:.4f}  → {c_old/J_lik:.0f}× likelihood')
print(f'Prior at start (new σ): {c_new:.4f}  → {c_new/J_lik:.2f}× likelihood')

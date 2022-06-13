# Context

This code and data was used to produce the modeling results presented in "Unsupervised Transient Light Curve Analysis Via Hierarchical Bayesian Inference" by Sanders, Betancourt, and Soderberg (arXiv:1404.3619).  It is closely related to the code used to generate the results in "Towards Characterization Of The Type IIp Supernova Progenitor Population: A Statistical Sample Of Light Curves From Pan-STARRS1" by Sanders, Soderberg, et al. (arXiv:1404.2004v2).

# Warning

This code was originally written in ~2013 and was cleaned up (a little) to share publicly in June 2022. The original code was not very well formatted or documented and I have made only limited efforts to improve it for public dissemination here.

The code is offered here without warrantee, though I would welcome people to leave GitHub issues if they have questions and I will do my best to respond.

# Components

All python code is for `python2`.  Unfortunately, I do not have a frozen record of the package environment originally used.  According to the original publication, [`CmdStan` version 2.2.0](https://github.com/stan-dev/stan/releases/tag/v2.2.0) was used for fitting, and/or an equivalent version of the `pystan` interface.

The primary code files are,

* `analyze.py` is the primary entrypoint to this code and generates all plots associated with the fitted Stan model.
* `utils.py` is a library of constant and function definitions referenced in the analysis file. The function `pystan_SPII_tm` is used for model fitting and the `runstan` wrapper does some associated data preprocessing and configuration. 
* `SIIP_t_multilevel_mod7_2.stan`: The `Stan` model code associated with the final model fit. The published version in the paper appendix should be considered the authoritative version; I believe this file is equivalent.

Some associated data files are,

* `allphot_galex_5.npy.tar.gz` is a gzipped serialized numpy array (with named columns) containing the processed Pan-STARRS1 Medium Deep Survey photometry modeled.
* `PS1_names.npy` and `PS1_shortnames.npy`: Serialized lists of Pan-STARRS object names, used for plotting.
* `PS1_lcfitpars_u3.npy`: Cached values of some light curve fitting parameter estimates, used in some plots.
* `PS1_ttype6.npy` and `PS1_ttype_u2.npy`: Some spectroscopic type information and other supernova metadata.

There is an additional file too large to be pushed to GitHub, which is therefore not included in this repo. This file is available upon request.

* `mod7_4.tar.gz` is a gzipped directory containing the `CmdStan` output traces for the fitted model. I believe these outputs were generated with, roughly, several parallel independent model fits generated with this command: `runstan(model='SIIP_t_multilevel_mod7.stan', nobjs='all', samples=50, chains=2, parallel=2, thin=1, sinit='0', delta=0.8, max_treedepth=18, startstep=1e-3)`

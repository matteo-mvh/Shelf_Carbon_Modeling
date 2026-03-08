"""Reduced DOC biology parameterizations following project documentation."""

from __future__ import annotations


def production_irradiance_pp(light, Pmax, K_L, n):
    """Hill-type PP(L) relationship [mol C m^-3]."""
    L = max(float(light), 0.0)
    exponent = max(float(n), 1e-12)
    hill_num = L**exponent
    hill_den = hill_num + max(float(K_L), 1e-12) ** exponent
    return float(Pmax) * hill_num / max(hill_den, 1e-12)


def remineralization_flux(ldoc, sdoc, rdoc, lambda_l, lambda_s, lambda_r):
    """Fremin = λL*LDOC + λS*SDOC + λR*RDOC [mol C m^-3 s^-1]."""
    return (
        float(lambda_l) * float(ldoc)
        + float(lambda_s) * float(sdoc)
        + float(lambda_r) * float(rdoc)
    )


def tendencies(
    light,
    ldoc,
    sdoc,
    rdoc,
    mu,
    alpha_l,
    alpha_s,
    alpha_r,
    lambda_l,
    lambda_s,
    lambda_r,
    gamma_l,
    gamma_s,
    Pmax,
    K_L,
    n,
):
    """DOC-pool tendencies and source/sink fluxes.

    Returns:
        dldoc_dt, dsdoc_dt, drdoc_dt, fprod, fremin
    """
    pp_light = production_irradiance_pp(light, Pmax=Pmax, K_L=K_L, n=n)
    fprod = float(mu) * pp_light

    ppl = float(alpha_l) * fprod
    pps = float(alpha_s) * fprod
    ppr = float(alpha_r) * fprod

    ldoc_v = float(ldoc)
    sdoc_v = float(sdoc)
    rdoc_v = float(rdoc)

    dldoc_dt = ppl - float(lambda_l) * ldoc_v - float(gamma_l) * ldoc_v
    dsdoc_dt = pps + float(gamma_l) * ldoc_v - float(lambda_s) * sdoc_v - float(gamma_s) * sdoc_v
    drdoc_dt = ppr + float(gamma_s) * sdoc_v - float(lambda_r) * rdoc_v

    fremin = remineralization_flux(
        ldoc_v,
        sdoc_v,
        rdoc_v,
        lambda_l=lambda_l,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
    )

    return dldoc_dt, dsdoc_dt, drdoc_dt, fprod, fremin

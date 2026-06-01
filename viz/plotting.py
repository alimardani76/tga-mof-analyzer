"""
TGA Visualization
==================
Publication-quality plots for TGA analysis results.

All functions return (fig, axes) so the user can further
customize before saving.

Dependencies: matplotlib (must be installed separately).
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


def _check_matplotlib():
    """Import matplotlib or raise a clear error."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'legend.fontsize': 10,
            'figure.dpi': 150,
        })
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting.\n"
            "Install it with: pip install matplotlib"
        )


# ------------------------------------------------------------------
# Plot 1: TGA + DTG dual-axis overlay
# ------------------------------------------------------------------

def plot_tga_dtg(
    tga_data,
    dtg_result=None,
    events=None,
    title: str = "TGA / DTG Analysis",
    show_events: bool = True,
    figsize: Tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Standard TGA + DTG overlay plot with optional event annotations.

    Left y-axis: mass (wt%)
    Right y-axis: DTG (%/°C)
    x-axis: temperature (°C)

    Parameters
    ----------
    tga_data : TGAData
    dtg_result : DTGResult or None
        If None, DTG is computed internally.
    events : EventDetectionResult or None
        If provided and show_events=True, events are shaded.
    title : str
    show_events : bool
    figsize : tuple
    save_path : str or None
        If given, saves the figure to this path.

    Returns
    -------
    (fig, (ax_tga, ax_dtg))
    """
    plt = _check_matplotlib()

    if dtg_result is None:
        from core.dtg import compute_dtg
        dtg_result = compute_dtg(tga_data)

    T = tga_data.temperature
    m = tga_data.mass_pct
    T_dtg = dtg_result.temperature
    dtg = dtg_result.dtg

    fig, ax1 = plt.subplots(figsize=figsize)

    # TGA curve (left axis)
    color_tga = '#2166ac'
    ax1.plot(T, m, color=color_tga, linewidth=1.5, label='TGA')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Mass (wt%)', color=color_tga)
    ax1.tick_params(axis='y', labelcolor=color_tga)
    ax1.set_ylim(bottom=max(0, m.min() - 5))

    # DTG curve (right axis)
    ax2 = ax1.twinx()
    color_dtg = '#b2182b'
    ax2.plot(T_dtg, dtg, color=color_dtg, linewidth=1.0,
             alpha=0.7, label='DTG')
    ax2.set_ylabel('DTG (%/°C)', color=color_dtg)
    ax2.tick_params(axis='y', labelcolor=color_dtg)
    ax2.set_ylim(bottom=0)

    # Event shading
    if show_events and events is not None:
        colors = plt.cm.Set3(np.linspace(0, 1, max(events.n_events, 1)))
        for i, ev in enumerate(events.events):
            ax1.axvspan(
                ev.t_start, ev.t_end,
                alpha=0.15, color=colors[i % len(colors)],
                label=f'E{i+1}: {ev.mass_loss_pct:.1f}%' if i < 8 else None,
            )

    # Annotations
    ax1.axhline(y=tga_data.residue_pct, color='gray', linestyle='--',
                linewidth=0.8, alpha=0.5)
    ax1.text(T[-1] * 0.95, tga_data.residue_pct + 1,
             f'Residue: {tga_data.residue_pct:.1f}%',
             ha='right', fontsize=9, color='gray')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=8, ncol=2)

    ax1.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, (ax1, ax2)


# ------------------------------------------------------------------
# Plot 2: R_exp(T) curve with plateau annotation
# ------------------------------------------------------------------

def plot_rexp_curve(
    tga_data,
    plateau_info: Optional[Dict] = None,
    title: str = "R_exp(T) Curve",
    figsize: Tuple = (10, 5),
    save_path: Optional[str] = None,
):
    """
    Plot R_exp as a function of temperature.

    Shows where the DH plateau is (or isn't), helping the user
    decide where to read the composition.

    The math:
        R_exp(T) = m(T) / m_residue

    A flat region means nothing is entering or leaving the MOF
    at that temperature.

    Parameters
    ----------
    tga_data : TGAData
    plateau_info : dict or None
        Output from find_dh_plateau(). If given, annotates
        the suggested plateau region.
    title : str
    figsize : tuple
    save_path : str or None

    Returns
    -------
    (fig, (ax_rexp, ax_drdt))
    """
    plt = _check_matplotlib()

    T = tga_data.temperature
    m = tga_data.mass_pct
    residue = tga_data.residue_pct

    r_exp = m / residue
    dr_dt = np.gradient(r_exp, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    height_ratios=[2, 1],
                                    sharex=True)

    # R_exp curve
    ax1.plot(T, r_exp, color='#2166ac', linewidth=1.5)
    ax1.set_ylabel('R_exp = m(T) / m_residue')
    ax1.set_title(title)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8,
                alpha=0.5, label='R_exp = 1 (pure residue)')
    ax1.legend(fontsize=9)

    # Plateau annotation
    if plateau_info is not None:
        t_lo, t_hi = plateau_info['suggested_range']
        r_val = plateau_info['r_exp_at_plateau']
        reliable = plateau_info['is_reliable']
        color = '#4daf4a' if reliable else '#ff7f00'
        ax1.axvspan(t_lo, t_hi, alpha=0.2, color=color,
                    label=f"Suggested DH plateau ({t_lo:.0f}–{t_hi:.0f}°C)")
        ax1.axhline(y=r_val, color=color, linestyle=':',
                    linewidth=1.0, alpha=0.8)
        ax1.text(t_hi + 5, r_val,
                 f'R_DH = {r_val:.3f}\n'
                 f'{"✅ reliable" if reliable else "⚠️ no true plateau"}',
                 fontsize=9, color=color, va='center')
        ax1.legend(fontsize=9)

    # dR/dT curve
    ax2.plot(T, dr_dt, color='#b2182b', linewidth=1.0)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('dR/dT (°C⁻¹)')
    ax2.set_ylim(bottom=min(dr_dt.min() * 1.1, -0.01),
                 top=max(dr_dt.max() * 1.1, 0.005))

    # Shade the "flat" region if plateau info given
    if plateau_info is not None:
        t_lo, t_hi = plateau_info['suggested_range']
        ax2.axvspan(t_lo, t_hi, alpha=0.2,
                    color='#4daf4a' if plateau_info['is_reliable'] else '#ff7f00')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, (ax1, ax2)


# ------------------------------------------------------------------
# Plot 3: Composition sensitivity (q vs T_DH)
# ------------------------------------------------------------------

def plot_composition_sensitivity(
    tga_data,
    components,
    t_range: Tuple[float, float] = (200, 500),
    n_points: int = 100,
    title: str = "Composition Sensitivity to DH Plateau Temperature",
    figsize: Tuple = (10, 5),
    save_path: Optional[str] = None,
):
    """
    Show how the computed linker count (q) depends on where
    you read the DH plateau temperature.

    This is the most important diagnostic for Module 3.
    It shows the user EXACTLY how sensitive the answer is
    to their choice of T_DH.

    The math at each temperature T:
        m_DH = interpolated mass at T
        R_exp_DH = m_DH / m_residue
        q = (R_exp_DH × M_residue − M_node) / M_linker

    Parameters
    ----------
    tga_data : TGAData
    components : MOFComponents
    t_range : tuple
        (T_min, T_max) for the sweep in °C.
    n_points : int
        Number of temperature points to evaluate.
    title : str
    figsize : tuple
    save_path : str or None

    Returns
    -------
    (fig, ax)
    """
    plt = _check_matplotlib()
    from core.rexp import compute_linkers

    T = tga_data.temperature
    m = tga_data.mass_pct
    residue = tga_data.residue_pct
    c = components

    t_sweep = np.linspace(
        max(t_range[0], T[0]),
        min(t_range[1], T[-1]),
        n_points,
    )

    q_values = []
    for t in t_sweep:
        m_at_t = float(np.interp(t, T, m))
        r_dh = m_at_t / residue
        q = compute_linkers(r_dh, c.M_residue, c.M_node, c.M_linker)
        q_values.append(q)

    q_values = np.array(q_values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t_sweep, q_values, color='#2166ac', linewidth=2.0)

    # Ideal linker line
    ideal_q = c.ideal_linkers
    ax.axhline(y=ideal_q, color='#4daf4a', linestyle='--',
               linewidth=1.5, label=f'Ideal q = {ideal_q:.0f}')

    # Plausible range shading
    ax.axhspan(ideal_q * 0.7, ideal_q * 1.05,
               alpha=0.1, color='#4daf4a',
               label=f'Plausible range ({ideal_q*0.7:.1f}–{ideal_q*1.05:.1f})')

    # Where q crosses ideal
    above = q_values > ideal_q
    if np.any(above) and np.any(~above):
        # Find the crossing point
        cross_idx = np.where(np.diff(above.astype(int)))[0]
        if len(cross_idx) > 0:
            t_cross = t_sweep[cross_idx[0]]
            ax.axvline(x=t_cross, color='gray', linestyle=':',
                       linewidth=1.0, alpha=0.5)
            ax.text(t_cross + 5, ideal_q * 0.5,
                    f'q = ideal at\nT = {t_cross:.0f}°C',
                    fontsize=9, color='gray')

    ax.set_xlabel('DH Plateau Temperature (°C)')
    ax.set_ylabel(f'Linkers per FU (q)')
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    # Add equation annotation
    eq_text = (
        f"q = (R_exp_DH × M_res − M_node) / M_linker\n"
        f"M_res = {c.M_residue:.1f},  M_node = {c.M_node:.1f},  "
        f"M_linker = {c.M_linker:.1f}"
    )
    ax.text(0.02, 0.02, eq_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


# ------------------------------------------------------------------
# Plot 4: Event map (color-coded TGA)
# ------------------------------------------------------------------

def plot_event_map(
    tga_data,
    events,
    title: str = "Event Map",
    figsize: Tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Color-coded TGA curve showing which temperature regions
    belong to which mass-loss event.

    Parameters
    ----------
    tga_data : TGAData
    events : EventDetectionResult
    title : str
    figsize : tuple
    save_path : str or None

    Returns
    -------
    (fig, ax)
    """
    plt = _check_matplotlib()

    T = tga_data.temperature
    m = tga_data.mass_pct

    fig, ax = plt.subplots(figsize=figsize)

    # Background TGA curve
    ax.plot(T, m, color='black', linewidth=0.5, alpha=0.3)

    # Color each event segment
    colors = plt.cm.tab20(np.linspace(0, 1, max(events.n_events, 1)))
    for i, ev in enumerate(events.events):
        mask = (T >= ev.t_start) & (T <= ev.t_end)
        if np.any(mask):
            ax.fill_between(
                T[mask], m[mask], ev.mass_end_pct,
                alpha=0.4, color=colors[i % len(colors)],
                label=f'E{i+1}: {ev.t_start:.0f}–{ev.t_end:.0f}°C '
                      f'({ev.mass_loss_pct:.1f}%)',
            )

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Mass (wt%)')
    ax.set_title(title)

    # Legend outside plot if many events
    if events.n_events > 8:
        ax.legend(fontsize=7, ncol=3, loc='upper right')
    else:
        ax.legend(fontsize=8, loc='best')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


# ------------------------------------------------------------------
# Plot 5: Composite loading series
# ------------------------------------------------------------------

def plot_composite_series(
    labels: List[str],
    mof_pcts: List[float],
    additive_name: str = "Additive",
    title: str = "Composite Loading",
    figsize: Tuple = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Bar chart of MOF loading across a series of composites.

    The math:
        w_MOF = (r_composite − r_additive) / (r_MOF − r_additive)
        where r = m_residue / m_initial

    Parameters
    ----------
    labels : list of str
        Sample labels.
    mof_pcts : list of float
        MOF weight fraction for each sample (%).
    additive_name : str
    title : str
    figsize : tuple
    save_path : str or None

    Returns
    -------
    (fig, ax)
    """
    plt = _check_matplotlib()

    add_pcts = [100 - m for m in mof_pcts]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, mof_pcts, width, label='MOF', color='#2166ac')
    ax.bar(x, add_pcts, width, bottom=mof_pcts,
           label=additive_name, color='#fdae61')

    ax.set_ylabel('Weight fraction (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)

    # Annotate MOF percentages
    for i, v in enumerate(mof_pcts):
        ax.text(i, v / 2, f'{v:.1f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Add equation
    eq_text = f"w_MOF = (r_comp − r_{additive_name}) / (r_MOF − r_{additive_name})"
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax
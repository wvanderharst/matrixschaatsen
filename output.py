import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _find_event_file(sim_folder, key_terms, suffix):
    for p in glob.glob(os.path.join(sim_folder, "*")):
        b = os.path.basename(p).lower()
        if all(k in b for k in key_terms) and b.endswith(suffix):
            return p
    return None

def _totals_from_df(df):
    if df is None or df.empty:
        return 0.0, 0.0
    # expect columns Win_Prob and Podium_Prob
    w = df.get('Win_Prob', pd.Series(dtype=float)).sum()
    p = df.get('Podium_Prob', pd.Series(dtype=float)).sum()
    return float(w), float(p)

def compare_women_1500(simulations_folder='simulations', save_name='women_1500_comparison.png'):
    sim_folder = simulations_folder
    os.makedirs(sim_folder, exist_ok=True)

    key_terms = ['1500', 'women']
    std_file = _find_event_file(sim_folder, key_terms, '_rider_stats_std.csv')
    nt_file  = _find_event_file(sim_folder, key_terms, '_rider_stats_no_top.csv')

    # fallback to aggregated slot files if per-event not found
    if std_file is None:
        agg_std = os.path.join(sim_folder, 'ned_slots_standard.csv')
        if os.path.exists(agg_std):
            df_std = pd.read_csv(agg_std)
            df_std = df_std[df_std['Event'].str.contains('1500', case=False, na=False) & (df_std['Gender'] == 'Women')]
        else:
            df_std = pd.DataFrame()
    else:
        df_std = pd.read_csv(std_file)

    if nt_file is None:
        agg_nt = os.path.join(sim_folder, 'ned_slots_no_top.csv')
        if os.path.exists(agg_nt):
            df_nt = pd.read_csv(agg_nt)
            df_nt = df_nt[df_nt['Event'].str.contains('1500', case=False, na=False) & (df_nt['Gender'] == 'Women')]
        else:
            df_nt = pd.DataFrame()
    else:
        df_nt = pd.read_csv(nt_file)

    win_std, pod_std = _totals_from_df(df_std)
    win_nt,  pod_nt  = _totals_from_df(df_nt)

    # pod_* values are sums of per-athlete podium probabilities => expected medal counts
    exp_medals_std = pod_std
    exp_medals_nt  = pod_nt

    # compute loss (no_top - std)
    win_loss = win_nt - win_std
    medal_loss = exp_medals_nt - exp_medals_std

    # Plot: clear, LinkedIn-friendly (1200x627) with two panels
    dpi = 100
    fig_w, fig_h = 1200 / dpi, 627 / dpi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi, gridspec_kw={'width_ratios':[1,1]})

    scen_names = ['Met Beune', 'Zonder Beune']
    colors = ['#2b8cbe', '#f03b20']

    # Left: total win probability (%)
    win_vals = [win_std * 100, win_nt * 100]
    x = np.arange(1)
    width = 0.35
    b1 = ax1.bar(x - width/2, win_vals[0], width, color=colors[0])
    b2 = ax1.bar(x + width/2, win_vals[1], width, color=colors[1])
    ax1.set_title('Goudkans (%)', fontsize=16, weight='bold', pad=10)
    ax1.set_xticks([])
    ax1.set_ylim(0, max(win_vals) * 1.25 + 1)
    for rect, val, name in zip([b1[0], b2[0]], win_vals, scen_names):
        ax1.text(rect.get_x() + rect.get_width()/2, val + 0.8, f"{val:.1f}%", ha='center', va='bottom', fontsize=12, weight='semibold')
        # place scenario label a fixed distance below the axis baseline (offset in points)
        ax1.annotate(name,
                     xy=(rect.get_x() + rect.get_width()/2, 0),
                     xycoords='data',
                     xytext=(0, -18), textcoords='offset points',
                     ha='center', va='top', fontsize=11, color='#333333')

    # Right: expected medals (count)
    medal_vals = [exp_medals_std, exp_medals_nt]
    b3 = ax2.bar(x - width/2, medal_vals[0], width, color=colors[0])
    b4 = ax2.bar(x + width/2, medal_vals[1], width, color=colors[1])
    ax2.set_title('Verwachte aantal medailles', fontsize=16, weight='bold', pad=10)
    ax2.set_xticks([])
    ax2.set_ylim(0, max(max(medal_vals), 1) * 1.5)
    for rect, val, name in zip([b3[0], b4[0]], medal_vals, scen_names):
        ax2.text(rect.get_x() + rect.get_width()/2, val + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=12, weight='semibold')
        # place scenario label a fixed distance below the axis baseline (offset in points)
        ax2.annotate(name,
                     xy=(rect.get_x() + rect.get_width()/2, 0),
                     xycoords='data',
                     xytext=(0, -18), textcoords='offset points',
                     ha='center', va='top', fontsize=11, color='#333333')

    # Overall title and subtitle (separate: increased spacing, smaller subtitle)
    title = "Goudkansen voor 1500 meter Vrouwen — Met Beune vs Zonder Beune"
    subtitle = "Kansen berekend uit simulaties van de Olympische Spelen gebaseerd op het model onderliggend aan de selectiematrix*"
    # main title
    fig.suptitle(title, fontsize=18, weight='bold', y=0.985)
    # subtitle placed lower (more gap) with smaller fontsize
    fig.text(0.5, 0.89, subtitle, ha='center', fontsize=12, color='#333333')

    # footer / source
    fig.text(0.99, 0.01, "Bron: Wouter van der Harst", ha='right', va='bottom', fontsize=8, color='#555555')
    fig.text(0.01, 0.01, "*Het gebruikte matrixmodel is niet publiek bekend. Mijn benadering is gebaseerd op het paper van Sierksma & Talsma (2021), de ontwikkelaars van de matrix", ha='left', va='bottom', fontsize=8, color='#555555')
    out_path = os.path.join(sim_folder, save_name)
    # keep layout tidy but reserve more space for title/subtitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    fig.savefig(out_path, dpi=dpi, facecolor='white')  # avoid bbox_inches='tight' to keep exact pixel size
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    path = compare_women_1500('simulations')
    print(f"Saved comparison image: {path}")

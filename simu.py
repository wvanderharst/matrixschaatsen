import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob

def run_all_skating_simulations(profiles_folder, simulations_folder, n_sims=10000):
    if not os.path.exists(simulations_folder):
        os.makedirs(simulations_folder)

    profile_files = glob.glob(os.path.join(profiles_folder, "*_profiles.csv"))
    slots_std, slots_dyn, slots_hyb, slots_no_top = [], [], [], []

    for file_path in profile_files:
        event_raw = os.path.basename(file_path).replace("_profiles.csv", "")
        event_name = event_raw.replace("_", " ")
        if any(x in event_name.lower() for x in ['mixed relay', 'team sprint', 'pursuit']): continue
            
        df = pd.read_csv(file_path)
        if df.empty: continue
        gender = 'Women' if 'Women' in event_name else 'Men'
        ned_df_full = df[df['Country'] == 'NED'].copy().reset_index(drop=True)
        non_ned_df = df[df['Country'] != 'NED'].copy().reset_index(drop=True)
        if len(ned_df_full) == 0: continue

        rows = np.arange(n_sims)
        intl_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in non_ned_df.iterrows()]).T
        ned_trial_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df_full.iterrows()]).T
        ned_final_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df_full.iterrows()]).T

        # --- 1. STANDARD ---
        ned_means = ned_df_full['Scale'] * np.exp((ned_df_full['Shape']**2) / 2)
        n_slots = min(3, len(ned_df_full))
        top_indices = np.argsort(ned_means.values)[:n_slots]
        field_std = np.hstack([ned_final_times[:, top_indices], intl_times])
        ranks_std = field_std.argsort(axis=1).argsort(axis=1) + 1
        
        event_win_prob_std = 0
        rider_stats_std = []
        for i, idx in enumerate(top_indices):
            w_p = np.sum(ranks_std[:, i] == 1) / n_sims
            p_p = np.sum(ranks_std[:, i] <= 3) / n_sims
            event_win_prob_std += w_p
            rider_stats_std.append({'Athlete': ned_df_full.iloc[idx]['Athlete'], 'Win_Prob': w_p, 'Podium_Prob': p_p})
            slots_std.append({'Event': event_name, 'Gender': gender, 'Slot': f"Top Potential {i+1}", 'Win_Prob': w_p, 'Podium_Prob': p_p})
        pd.DataFrame(rider_stats_std).to_csv(os.path.join(simulations_folder, f"{event_raw}_rider_stats_std.csv"), index=False)

        # --- 2. DYNAMIC ---
        q_dyn = np.argsort(ned_trial_times, axis=1)[:, :n_slots]
        field_dyn = np.hstack([ned_final_times[rows[:, None], q_dyn], intl_times])
        ranks_dyn = field_dyn.argsort(axis=1).argsort(axis=1) + 1
        
        rider_stats_dyn = []
        for idx, row in ned_df_full.iterrows():
            mask = np.any(q_dyn == idx, axis=1)
            if np.any(mask):
                pos = (q_dyn[mask] == idx).argmax(axis=1)
                finishes = ranks_dyn[mask, pos]
                rider_stats_dyn.append({'Athlete': row['Athlete'], 'Qualify_Prob': np.mean(mask), 
                                        'Win_Prob': np.sum(finishes == 1) / n_sims, 'Podium_Prob': np.sum(finishes <= 3) / n_sims})
            else:
                rider_stats_dyn.append({'Athlete': row['Athlete'], 'Qualify_Prob': 0, 'Win_Prob': 0, 'Podium_Prob': 0})
        for i in range(n_slots):
            slots_dyn.append({'Event': event_name, 'Gender': gender, 'Slot': f"Trial Rank {i+1}", 'Win_Prob': np.sum(ranks_dyn[:, i] == 1) / n_sims, 'Podium_Prob': np.sum(ranks_dyn[:, i] <= 3) / n_sims})
        pd.DataFrame(rider_stats_dyn).to_csv(os.path.join(simulations_folder, f"{event_raw}_rider_stats_dyn.csv"), index=False)

        # --- 3. HYBRID ---
        std_win_map = {r['Athlete']: r['Win_Prob'] for r in rider_stats_std}
        prot_idx = sorted([j for j in range(len(ned_df_full)) if std_win_map.get(ned_df_full.iloc[j]['Athlete'], 0) > 0.10], 
                          key=lambda x: std_win_map.get(ned_df_full.iloc[x]['Athlete'], 0), reverse=True)[:2]
        ned_trial_hyb = ned_trial_times.copy()
        for pi in prot_idx: ned_trial_hyb[:, pi] = 999.9
        trial_winners = np.argsort(ned_trial_hyb, axis=1)[:, : (n_slots - len(prot_idx))]
        q_hyb = np.hstack([np.full((n_sims, len(prot_idx)), prot_idx), trial_winners]).astype(int)
        field_hyb = np.hstack([ned_final_times[rows[:, None], q_hyb], intl_times])
        ranks_hyb = field_hyb.argsort(axis=1).argsort(axis=1) + 1
        
        rider_stats_hyb = []
        for idx, row in ned_df_full.iterrows():
            mask = np.any(q_hyb == idx, axis=1)
            if np.any(mask):
                pos = (q_hyb[mask] == idx).argmax(axis=1)
                finishes = ranks_hyb[mask, pos]
                rider_stats_hyb.append({'Athlete': row['Athlete'], 'Qualify_Prob': np.mean(mask), 
                                        'Win_Prob': np.sum(finishes == 1) / n_sims, 'Podium_Prob': np.sum(finishes <= 3) / n_sims})
            else:
                rider_stats_hyb.append({'Athlete': row['Athlete'], 'Qualify_Prob': 0, 'Win_Prob': 0, 'Podium_Prob': 0})
        for i in range(n_slots):
            lbl = f"Protected {i+1}" if i < len(prot_idx) else f"Trial Winner {i - len(prot_idx) + 1}"
            slots_hyb.append({'Event': event_name, 'Gender': gender, 'Slot': lbl, 'Win_Prob': np.sum(ranks_hyb[:, i] == 1) / n_sims, 'Podium_Prob': np.sum(ranks_hyb[:, i] <= 3) / n_sims})
        pd.DataFrame(rider_stats_hyb).to_csv(os.path.join(simulations_folder, f"{event_raw}_rider_stats_hyb.csv"), index=False)

        # --- 4. NO TOP ATHLETE + LOSS CALCULATION ---
        top_athlete_idx = np.argmin(ned_means.values)
        top_athlete_name = ned_df_full.iloc[top_athlete_idx]['Athlete']
        ned_df_nt = ned_df_full.drop(top_athlete_idx).reset_index(drop=True)
        n_slots_nt = min(3, len(ned_df_nt))

        if n_slots_nt > 0:
            trial_nt = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df_nt.iterrows()]).T
            final_nt = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df_nt.iterrows()]).T
            q_nt = np.argsort(trial_nt, axis=1)[:, :n_slots_nt]
            field_nt = np.hstack([final_nt[rows[:, None], q_nt], intl_times])
            ranks_nt = field_nt.argsort(axis=1).argsort(axis=1) + 1
            
            event_win_prob_nt = 0
            rider_stats_nt = []
            for idx, row in ned_df_nt.iterrows():
                mask = np.any(q_nt == idx, axis=1)
                if np.any(mask):
                    pos = (q_nt[mask] == idx).argmax(axis=1)
                    finishes = ranks_nt[mask, pos]
                    w_p = np.sum(finishes == 1) / n_sims
                    event_win_prob_nt += w_p
                    rider_stats_nt.append({'Athlete': row['Athlete'], 'Qualify_Prob': np.mean(mask), 'Win_Prob': w_p, 'Podium_Prob': np.sum(finishes <= 3) / n_sims})
                else:
                    rider_stats_nt.append({'Athlete': row['Athlete'], 'Qualify_Prob': 0, 'Win_Prob': 0, 'Podium_Prob': 0})
            
            # Bereken het verlies in winstkans voor deze afstand
            event_loss = event_win_prob_nt - event_win_prob_std
            
            for i in range(n_slots_nt):
                slots_no_top.append({
                    'Event': event_name, 'Gender': gender, 'Slot': f"Sub-Top Trial {i+1}", 
                    'Win_Prob': np.sum(ranks_nt[:, i] == 1) / n_sims, 
                    'Podium_Prob': np.sum(ranks_nt[:, i] <= 3) / n_sims,
                    'Excluded_Athlete': top_athlete_name,
                    'Event_Win_Loss_vs_Std': round(event_loss, 4)
                })
            pd.DataFrame(rider_stats_nt).to_csv(os.path.join(simulations_folder, f"{event_raw}_rider_stats_no_top.csv"), index=False)

    # --- FINALIZE ---
    def finalize(data, name, sort_cols=['Gender', 'Win_Prob'], asc=[True, False]):
        df_res = pd.DataFrame(data)
        if not df_res.empty:
            df_res = df_res.sort_values(sort_cols, ascending=asc)
            df_res.to_csv(name, index=False)
        return df_res

    finalize(slots_std, 'ned_slots_standard.csv')
    finalize(slots_dyn, 'ned_slots_dynamic.csv')
    finalize(slots_hyb, 'ned_slots_hybrid.csv')
    # Sorteren op verlies (grootste negatieve impact bovenaan)
    finalize(slots_no_top, 'ned_slots_no_top.csv', sort_cols=['Gender', 'Event_Win_Loss_vs_Std'], asc=[True, True])

    print("\nSimulatie voltooid. Check 'ned_slots_no_top.csv' voor de winstkans-verliezen per afstand.")

if __name__ == "__main__":
    run_all_skating_simulations('distributionfitted', 'simulations')
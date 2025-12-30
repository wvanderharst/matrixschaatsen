import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob

import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob

def run_comprehensive_athlete_rankings(profiles_folder, simulations_folder, n_sims=10000):
    if not os.path.exists(simulations_folder):
        os.makedirs(simulations_folder)

    profile_files = glob.glob(os.path.join(profiles_folder, "*_profiles.csv"))
    
    all_std_stats = []
    all_dyn_stats = []
    all_hyb_stats = []

    for file_path in profile_files:
        event_raw = os.path.basename(file_path).replace("_profiles.csv", "")
        event_name = event_raw.replace("_", " ")
        if any(x in event_name.lower() for x in ['mixed relay', 'team sprint', 'pursuit']): continue
            
        df = pd.read_csv(file_path)
        if df.empty: continue
        gender = 'Women' if 'Women' in event_name else 'Men'
        ned_df = df[df['Country'] == 'NED'].copy().reset_index(drop=True)
        non_ned_df = df[df['Country'] != 'NED'].copy().reset_index(drop=True)
        if ned_df.empty: continue

        print(f"Ranking Athletes for {event_name}...")

        # --- PRE-GENERATE DATA ---
        all_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in df.iterrows()]).T
        ned_trial_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df.iterrows()]).T
        ned_final_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in ned_df.iterrows()]).T
        intl_times = np.array([stats.lognorm.rvs(s=r['Shape'], scale=r['Scale'], size=n_sims) for _, r in non_ned_df.iterrows()]).T
        rows = np.arange(n_sims)

        # --- MODE 1: STANDARD ---
        ranks_std = all_times.argsort(axis=1).argsort(axis=1) + 1
        for i, row in df[df['Country'] == 'NED'].iterrows():
            orig_idx = df[df['Athlete'] == row['Athlete']].index[0]
            finishes = ranks_std[:, orig_idx]
            all_std_stats.append({
                'Athlete': row['Athlete'], 'Event': event_name, 'Gender': gender,
                'Avg_Finish': round(np.mean(finishes), 3), 'Win_Prob': round(np.sum(finishes == 1) / n_sims, 4)
            })

        # --- MODE 2: DYNAMIC ---
        q_dyn = np.argsort(ned_trial_times, axis=1)[:, :3]
        field_dyn = np.hstack([ned_final_times[rows[:, None], q_dyn], intl_times])
        ranks_dyn = field_dyn.argsort(axis=1).argsort(axis=1) + 1
        
        for idx, row in ned_df.iterrows():
            mask = np.any(q_dyn == idx, axis=1)
            if np.any(mask):
                # Find which column (0, 1, or 2) the athlete is in for each successful simulation
                pos_in_q = (q_dyn[mask] == idx).argmax(axis=1)
                finishes = ranks_dyn[mask, pos_in_q]
                all_dyn_stats.append({
                    'Athlete': row['Athlete'], 'Event': event_name, 'Gender': gender,
                    'Qualify_Prob': round(np.mean(mask), 3),
                    'Avg_Finish_When_Qualified': round(np.mean(finishes), 3)
                })

        # --- MODE 3: HYBRID ---
        # Get standard win probs to determine protection
        std_win_probs = [np.sum(ranks_std[:, df[df['Athlete']==r['Athlete']].index[0]] == 1) / n_sims for _, r in ned_df.iterrows()]
        protected_indices = [i for i, prob in enumerate(std_win_probs) if prob > 0.10]
        protected_indices = sorted(protected_indices, key=lambda x: std_win_probs[x], reverse=True)[:2]
        
        ned_trial_copy = ned_trial_times.copy()
        for pi in protected_indices: ned_trial_copy[:, pi] = 999.9 # Exclude protected from trial
        
        num_to_fill = 3 - len(protected_indices)
        trial_winners = np.argsort(ned_trial_copy, axis=1)[:, :num_to_fill]
        q_hyb = np.hstack([np.full((n_sims, len(protected_indices)), protected_indices), trial_winners]).astype(int)
        
        field_hyb = np.hstack([ned_final_times[rows[:, None], q_hyb], intl_times])
        ranks_hyb = field_hyb.argsort(axis=1).argsort(axis=1) + 1

        for idx, row in ned_df.iterrows():
            mask = np.any(q_hyb == idx, axis=1)
            if np.any(mask):
                # Use argmax to find the column index where the athlete exists in q_hyb
                pos_in_q = (q_hyb[mask] == idx).argmax(axis=1)
                finishes = ranks_hyb[mask, pos_in_q]
                all_hyb_stats.append({
                    'Athlete': row['Athlete'], 'Event': event_name, 'Gender': gender,
                    'Protected': (idx in protected_indices),
                    'Avg_Finish_In_Hybrid': round(np.mean(finishes), 3)
                })

    # --- SAVE ---
    def save_ranked(data, filename, sort_col):
        res = pd.DataFrame(data)
        if not res.empty:
            res['Rank_in_Gender'] = res.groupby('Gender')[sort_col].rank(method='min')
            res.sort_values(['Gender', 'Rank_in_Gender']).to_csv(filename, index=False)

    save_ranked(all_std_stats, 'ned_individual_rankings_standard.csv', 'Avg_Finish')
    save_ranked(all_dyn_stats, 'ned_individual_rankings_dynamic.csv', 'Avg_Finish_When_Qualified')
    save_ranked(all_hyb_stats, 'ned_individual_rankings_hybrid.csv', 'Avg_Finish_In_Hybrid')
    print("\nFixed shape mismatch. Athlete CSVs created.")

run_comprehensive_athlete_rankings('distributionfitted', 'simulations')
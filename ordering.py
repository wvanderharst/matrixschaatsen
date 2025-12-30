import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob

def run_ranked_simulations(profiles_folder, n_sims=5000):
    # Containers for data
    summary_no_sel, summary_with_sel = [], []
    ranks_no_sel, ranks_with_sel = [], []
    trial_performance_stats = []
    
    profile_files = glob.glob(os.path.join(profiles_folder, "*_individual_fits.csv"))
    
    for file_path in profile_files:
        event_raw = os.path.basename(file_path).replace("_individual_fits.csv", "")
        event_display_name = event_raw.replace("__", " ").replace("_", " ")
        
        # Exclusions
        if any(x.lower() in event_display_name.lower() for x in ['mixed relay', 'team sprint']):
            continue
            
        df = pd.read_csv(file_path)
        df = df[df['Scale'] > 0].copy() # Ensure valid distribution
        if df.empty: continue
        
        gender = 'Women' if 'Women' in event_display_name else 'Men'
        ned_df = df[df['Country'] == 'NED'].copy()
        non_ned_df = df[df['Country'] != 'NED'].copy()
        if ned_df.empty: continue

        # --- MODE 1: NO SELECTION ---
        all_countries = df['Country'].values
        sim_all = np.zeros((n_sims, len(df)))
        for i, (_, row) in enumerate(df.iterrows()):
            sim_all[:, i] = stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims)
        
        ranks_all = np.zeros_like(sim_all)
        for s in range(n_sims):
            ranks_all[s] = sim_all[s].argsort().argsort() + 1
            
        ned_mask_all = all_countries == 'NED'
        ned_ranks_all_sorted = np.sort(ranks_all[:, ned_mask_all], axis=1)
        for order in range(min(ned_ranks_all_sorted.shape[1], 3)):
            ranks_no_sel.append({
                'Event': event_display_name, 'Gender': gender,
                'NED_Slot': f"NED Rider {order + 1}", 'Avg_Placing': round(np.mean(ned_ranks_all_sorted[:, order]), 3)
            })
            
        gold_all_prob = np.sum(all_countries[sim_all.argmin(axis=1)] == 'NED') / n_sims
        summary_no_sel.append({'Event': event_display_name, 'Gender': gender, 'NED_Gold_Prob': gold_all_prob})

        # --- MODE 2: WITH SELECTION (Trial-based) ---
        ned_df['Theor_Mean'] = ned_df['Scale'] * np.exp((ned_df['Shape']**2) / 2)
        top3_qualified = ned_df.sort_values('Theor_Mean').head(3).copy()
        top3_qualified['Trial_Status'] = [f"Trial Rider {i+1}" for i in range(len(top3_qualified))]
        
        final_field = pd.concat([top3_qualified, non_ned_df]).reset_index(drop=True)
        final_countries = final_field['Country'].values
        sim_final = np.zeros((n_sims, len(final_field)))
        for i, (_, row) in enumerate(final_field.iterrows()):
            sim_final[:, i] = stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims)
            
        ranks_final = np.zeros_like(sim_final)
        for s in range(n_sims):
            ranks_final[s] = sim_final[s].argsort().argsort() + 1
            
        # TRACKING: Trial Rider Performance (Specific Individuals)
        for idx in np.where(final_countries == 'NED')[0]:
            trial_performance_stats.append({
                'Event': event_display_name, 'Gender': gender,
                'Trial_Status': final_field.loc[idx, 'Trial_Status'],
                'Athlete': final_field.loc[idx, 'Athlete'],
                'Avg_Finish_Position': round(np.mean(ranks_final[:, idx]), 3)
            })
            
        # TRACKING: Order Statistics (Slots)
        ned_ranks_final_sorted = np.sort(ranks_final[:, final_countries == 'NED'], axis=1)
        for order in range(ned_ranks_final_sorted.shape[1]):
            ranks_with_sel.append({
                'Event': event_display_name, 'Gender': gender,
                'NED_Slot': f"NED Rider {order + 1}", 'Avg_Placing': round(np.mean(ned_ranks_final_sorted[:, order]), 3)
            })
            
        gold_final_prob = np.sum(final_countries[sim_final.argmin(axis=1)] == 'NED') / n_sims
        summary_with_sel.append({'Event': event_display_name, 'Gender': gender, 'NED_Gold_Prob': gold_final_prob})

    # --- RANKING AND SAVING ---
    def save_and_rank(data, filename, sort_col, ascending=True):
        df_out = pd.DataFrame(data)
        if not df_out.empty:
            # Add Rank within each Gender group
            df_out['Rank_in_Gender'] = df_out.groupby('Gender')[sort_col].rank(method='min', ascending=ascending)
            # Final Sort for the CSV
            df_out = df_out.sort_values(['Gender', 'Rank_in_Gender'])
            df_out.to_csv(filename, index=False)
            print(f"Generated: {filename}")

    save_and_rank(summary_no_sel, 'ned_summary_NO_selection_ranked.csv', 'NED_Gold_Prob', False)
    save_and_rank(summary_with_sel, 'ned_summary_WITH_selection_ranked.csv', 'NED_Gold_Prob', False)
    save_and_rank(ranks_no_sel, 'ned_rankings_NO_selection_ranked.csv', 'Avg_Placing', True)
    save_and_rank(ranks_with_sel, 'ned_rankings_WITH_selection_ranked.csv', 'Avg_Placing', True)
    save_and_rank(trial_performance_stats, 'ned_trial_rider_performance_ranked.csv', 'Avg_Finish_Position', True)

# Run the simulation
run_ranked_simulations('distributionfitted')
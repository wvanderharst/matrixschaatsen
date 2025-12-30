import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob


def run_fast_dynamic_selection(profiles_folder, output_file, n_sims=10000):
    """
    Optimized version using NumPy vectorization to speed up simulations.
    Final output is ordered by Gender and Average Finish Position.
    """
    profile_files = glob.glob(os.path.join(profiles_folder, "*_profiles.csv"))
    final_results = []

    for file_path in profile_files:
        event_raw = os.path.basename(file_path).replace("_profiles.csv", "")
        event_display_name = event_raw.replace("_", " ")
        
        if any(x in event_display_name.lower() for x in ['mixed relay', 'team sprint', 'pursuit']):
            continue
            
        df = pd.read_csv(file_path)
        if df.empty: continue
            
        ned_df = df[df['Country'] == 'NED'].copy().reset_index(drop=True)
        non_ned_df = df[df['Country'] != 'NED'].copy().reset_index(drop=True)
        if ned_df.empty: continue

        print(f"Simulating {event_display_name}...")

        # --- PHASE 1: VECTORIZED DUTCH TRIALS ---
        # Generate 5000 trial times for all NED riders at once
        ned_trial_times = np.array([
            stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims) 
            for _, row in ned_df.iterrows()
        ]).T
        
        # Get indices of top 3 in each of the 5000 trials
        trial_ranks = np.argsort(ned_trial_times, axis=1)
        trial_winners = trial_ranks[:, 0]
        trial_seconds = trial_ranks[:, 1]
        trial_thirds  = trial_ranks[:, 2] if trial_ranks.shape[1] > 2 else trial_ranks[:, 1]

        # --- PHASE 2: VECTORIZED INTERNATIONAL FINAL ---
        ned_final_times = np.array([
            stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims) 
            for _, row in ned_df.iterrows()
        ]).T
        
        intl_final_times = np.array([
            stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims) 
            for _, row in non_ned_df.iterrows()
        ]).T

        # Extract times for the 3 qualified slots
        rows = np.arange(n_sims)
        q1_times = ned_final_times[rows, trial_winners].reshape(-1, 1)
        q2_times = ned_final_times[rows, trial_seconds].reshape(-1, 1)
        q3_times = ned_final_times[rows, trial_thirds].reshape(-1, 1)
        
        # Combine into international field
        full_final_field = np.hstack([q1_times, q2_times, q3_times, intl_final_times])
        
        # Rank the final field (Smallest time = Rank 1)
        final_ranks = full_final_field.argsort(axis=1).argsort(axis=1) + 1
        
        # --- COLLECT STATS ---
        gender = 'Women' if 'Women' in event_display_name else 'Men'
        for i in range(3):
            finishes = final_ranks[:, i]
            final_results.append({
                'Event': event_display_name,
                'Gender': gender,
                'Selection_Result': f"Trial Rank {i+1}",
                'Avg_Final_Finish_Pos': round(np.mean(finishes), 3),
                'Win_Prob_In_Final': round(np.sum(finishes == 1) / n_sims, 4)
            })

    # --- SAVE AND ORDER ---
    out_df = pd.DataFrame(final_results)
    if not out_df.empty:
        # Ordering the file: First by Gender, then by the best average finish
        out_df = out_df.sort_values(by=['Gender', 'Win_Prob_In_Final'], ascending=[True, False])
        
        out_df.to_csv(output_file, index=False)
        print(f"\nDone! Results saved to {output_file} (Ordered by Gender and Finish Position)")

# Run
run_fast_dynamic_selection('distributionfitted', 'ned_dynamic_selection_results.csv')

# --- ORIGINAL FUNCTION ---
def run_simulations_to_folder(profiles_folder, simulations_folder, summary_file, n_sims=10000):
    if not os.path.exists(simulations_folder):
        os.makedirs(simulations_folder)
    all_event_summaries = []
    profile_files = glob.glob(os.path.join(profiles_folder, "*_profiles.csv"))
    for file_path in profile_files:
        event_filename = os.path.basename(file_path).replace("_profiles.csv", "")
        event_display_name = event_filename.replace("_", " ")
        df = pd.read_csv(file_path)
        if df.empty: continue
        n_riders = len(df)
        sim_results = np.zeros((n_sims, n_riders))
        for i, row in df.iterrows():
            sim_results[:, i] = stats.lognorm.rvs(s=row['Shape'], scale=row['Scale'], size=n_sims)
        ranks = sim_results.argsort(axis=1)
        gold_idx = ranks[:, 0]
        rider_event_stats = []
        unique_winners, counts = np.unique(gold_idx, return_counts=True)
        win_map = dict(zip(unique_winners, counts))
        unique_podium, p_counts = np.unique(ranks[:, :3], return_counts=True)
        podium_map = dict(zip(unique_podium, p_counts))
        for i, row in df.iterrows():
            wins = int(win_map.get(i, 0))
            podiums = int(podium_map.get(i, 0))
            rider_event_stats.append({
                'Athlete': row['Athlete'], 'Country': row['Country'],
                'Wins': wins, 'Win_Prob': wins / n_sims,
                'Podiums': podiums, 'Podium_Prob': podiums / n_sims
            })
        rider_event_df = pd.DataFrame(rider_event_stats).sort_values('Wins', ascending=False)
        rider_event_df.to_csv(os.path.join(simulations_folder, f"{event_filename}_rider_stats.csv"), index=False)
        countries = df['Country'].values
        ned_gold = int(np.sum(countries[gold_idx] == 'NED'))
        ned_silver = int(np.sum(countries[ranks[:, 1]] == 'NED'))
        ned_bronze = int(np.sum(countries[ranks[:, 2]] == 'NED'))
        all_event_summaries.append({
            'Event': event_display_name, 'NED_Gold': ned_gold,
            'NED_Silver': ned_silver, 'NED_Bronze': ned_bronze,
            'NED_Total_Medals': ned_gold + ned_silver + ned_bronze
        })
    summary_df = pd.DataFrame(all_event_summaries).sort_values('NED_Gold', ascending=False)
    summary_df.to_csv(summary_file, index=False)
    print(f"Simulation files created in '{simulations_folder}/'")

# --- RUN BOTH ---
# 1. Standard Simulation (everyone starts)
run_simulations_to_folder('distributionfitted', 'simulations', 'ned_medal_summary.csv')
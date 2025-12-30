import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def create_rider_profiles(base_input_folder, output_folder):
    """
    Reads event CSVs, shifts values, fits Log-Normal distributions, 
    imputes shape for low-obs skaters, and sorts from best to worst.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the event subfolders created by your first script
    for root, dirs, files in os.walk(base_input_folder):
        for file in files:
            if not file.endswith(".csv"): continue
            
            file_path = os.path.join(root, file)
            event_name = file.replace(".csv", "")
            df = pd.read_csv(file_path)
            
            # Identify date columns
            date_cols = [c for c in df.columns if c not in ['Athlete', 'Country']]
            
            # 1. FIND GLOBAL MINIMUM AND SHIFT
            global_min = df[date_cols].min().min()
            shift_constant = abs(global_min) + 0.1
            
            # 2. CALCULATE INDIVIDUAL STATS
            rider_stats = []
            valid_sigmas = []

            for _, row in df.iterrows():
                # Extract non-null shifted diffs
                vals = row[date_cols].values.astype(float)
                vals = vals[~np.isnan(vals)] + shift_constant
                
                n_obs = len(vals)
                mean_val = np.mean(vals)
                
                sigma = None
                if n_obs >= 3:
                    try:
                        # 2-parameter fit (fixing location at 0)
                        shape, loc, scale = stats.lognorm.fit(vals, floc=0)
                        sigma = shape
                        valid_sigmas.append(shape)
                    except:
                        pass
                
                rider_stats.append({
                    'Athlete': row['Athlete'],
                    'Country': row['Country'],
                    'Races': n_obs,
                    'Shifted_Mean': mean_val,
                    'Shape': sigma,
                    'Shift_Used': shift_constant
                })

            # 3. GLOBAL SHAPE IMPUTATION
            avg_event_shape = np.mean(valid_sigmas) if valid_sigmas else 0.2
            
            final_profiles = []
            for rider in rider_stats:
                if rider['Shape'] is None:
                    rider['Shape'] = avg_event_shape
                    rider['Profile_Type'] = 'Synthetic'
                else:
                    rider['Profile_Type'] = 'Actual'
                
                # mu = ln(mean) - (sigma^2 / 2)
                mu = np.log(rider['Shifted_Mean']) - (rider['Shape']**2 / 2)
                rider['Scale'] = np.exp(mu)
                final_profiles.append(rider)

            # 4. SORTING: Best to Worst
            # Best skaters have the lowest Shifted_Mean
            profile_df = pd.DataFrame(final_profiles)
            profile_df = profile_df.sort_values(by='Shifted_Mean', ascending=True)

            # 5. SAVE PROFILES
            output_path = os.path.join(output_folder, f"{event_name}_profiles.csv")
            profile_df.to_csv(output_path, index=False)
            print(f"Exported Sorted Profiles: {event_name}")

# --- SETTINGS ---
DISTRIBUTION_INPUT = 'distributiefitting'
PROFILES_OUTPUT = 'distributionfitted'

if __name__ == "__main__":
    create_rider_profiles(DISTRIBUTION_INPUT, PROFILES_OUTPUT)
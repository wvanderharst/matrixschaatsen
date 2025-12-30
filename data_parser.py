import pandas as pd
import re
import os
import glob
import numpy as np
import scipy.stats as stats

def time_to_seconds(time_str):
    """Converts speed skating time strings (MM:SS.ms or SS.ms) to total seconds."""
    if not time_str or time_str in ['DNF', 'DQ', 'DNS', 'WDR', 'MT']:
        return None
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return None

def clean_name(text):
    """Makes a string safe for folder and file names."""
    return re.sub(r'[\\/*?:"<>|]', "", text).strip()

def process_and_save_by_event_folders(input_path, base_output_folder):
    all_data = []
    current_event, current_date = "Unknown", "Unknown"
    event_pattern = re.compile(r'^(.+),\s*(\d{2}\.\d{2}\.\d{4})')
    
    # Identify files to process (handles folder or single file)
    files = glob.glob(os.path.join(input_path, "*.txt")) if os.path.isdir(input_path) else [input_path]
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('POS'): continue
                
                event_match = event_pattern.match(line)
                if event_match:
                    current_event, current_date = event_match.group(1).strip(), event_match.group(2).strip()
                    continue
                
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 5:
                    # Handle potential merging of Rank and Name
                    first_part = parts[0].split(None, 1)
                    if len(first_part) > 1 and (first_part[0].isdigit() or first_part[0] in ['DNF', 'DQ', 'DNS']):
                        pos, name = first_part[0], first_part[1]
                        remaining = parts[1:]
                    else:
                        pos, name = parts[0], parts[1]
                        remaining = parts[2:]
                    
                    if len(remaining) >= 3:
                        all_data.append({
                            'Event': current_event, 
                            'Date': current_date, 
                            'Rank': pos,
                            'Athlete': name, 
                            'Country': remaining[1], # Nationality
                            'Time_Str': remaining[2]
                        })

    if not all_data: return
    df = pd.DataFrame(all_data)
    
    # Filter 1: Remove DNF, DNS, DQ (Keep only numeric ranks)
    df = df[df['Rank'].str.isdigit()].copy()
    df['Time_Sec'] = df['Time_Str'].apply(time_to_seconds)
    df['Rank_Num'] = pd.to_numeric(df['Rank'])
    df = df.dropna(subset=['Time_Sec'])

    def apply_filters(group):
        event_name = group['Event'].iloc[0].lower()
        # Mass Start Logic: Remove athletes ranked lower than winner with faster times
        if 'mass start' in event_name:
            winner_time = group.loc[group['Rank_Num'] == 1, 'Time_Sec']
            if not winner_time.empty:
                group = group[~((group['Rank_Num'] > 1) & (group['Time_Sec'] < winner_time.iloc[0]))]
        
        # Fall Filter: Remove times > 20% slower than fastest
        fastest = group['Time_Sec'].min()
        group = group[group['Time_Sec'] <= (fastest * 1.20)].copy()
        
        # Calculate Top 5 Average baseline
        group['Avg_Top_5'] = group.sort_values('Rank_Num')['Time_Sec'].head(5).mean()
        group['Diff'] = group['Time_Sec'] - group['Avg_Top_5']
        return group

    # Apply calculations per specific race (Event + Date)
    df = df.groupby(['Event', 'Date'], group_keys=False).apply(apply_filters)

    # Save to directory structure
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    for event_name, event_group in df.groupby('Event'):
        event_dir = os.path.join(base_output_folder, clean_name(event_name))
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)
        
        # Pivot so Rows = Athlete/Country and Columns = Dates
        # AgeGroup is dropped here to avoid duplicate rows for the same athlete
        pivot_df = event_group.pivot_table(
            index=['Athlete', 'Country'],
            columns='Date',
            values='Diff'
        ).reset_index()
        
        filename = f"{clean_name(event_name)}.csv"
        pivot_df.to_csv(os.path.join(event_dir, filename), index=False)
        print(f"Exported: {event_name}")

# --- SETTINGS ---
INPUT_PATH = r'c:\Users\woute\Documents\Projects\matrixschaatsen\raw_data'
OUTPUT_BASE = 'distributiefitting'

if __name__ == "__main__":
    process_and_save_by_event_folders(INPUT_PATH, OUTPUT_BASE)
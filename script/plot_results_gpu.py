import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'legend.fontsize': 10
})

CSV_FILE = "benchmark_results_gpu_7.0_v3.csv" 

def parse_stress_metadata(stressor_name):
    """Estrae la categoria di stress e il numero di worker come feature numerica"""
    if pd.isna(stressor_name) or not isinstance(stressor_name, str):
        return "Unknown", 0
    if stressor_name == "Baseline_NoStress":
        return "No Stress", 0
    
    match = re.match(r"(.+)_W(\d+)", stressor_name)
    if match:
        category = match.group(1).replace("_", " ")
        workers = int(match.group(2))
        return category, workers
    return stressor_name, 0

def generate_degradation_curves(df, temp, filename):
    """Genera le curve di degradazione sfruttando le 10 ripetizioni per mostrare la varianza (bande di errore)"""
    df_temp = df[df['Temperatura'] == str(temp)].copy()
    
    # Separiamo le baseline dagli stress (solo a W=4)
    baseline_df = df_temp[df_temp['Stress_Category'] == "No Stress"]
    stress_df = df_temp[df_temp['Stress_Category'] != "No Stress"]
    
    plot_data = []
    categories = stress_df['Stress_Category'].unique()
    
    for cat in categories:
        # Aggiungiamo TUTTE e 10 le run di baseline associandole a questa categoria (a Worker=0)
        for _, row in baseline_df.iterrows():
            plot_data.append({
                'Stress_Category': cat, 
                'Workers': 0, 
                'Generation_Throughput_ts': row['Generation_Throughput_ts']
            })
            
        # Aggiungiamo TUTTE e 10 le run sotto stress (a Worker=4)
        sub_df = stress_df[stress_df['Stress_Category'] == cat]
        for _, row in sub_df.iterrows():
            plot_data.append({
                'Stress_Category': cat, 
                'Workers': row['Workers'], 
                'Generation_Throughput_ts': row['Generation_Throughput_ts']
            })
            
    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.sort_values(by=['Stress_Category', 'Workers'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Workers', y='Generation_Throughput_ts', hue='Stress_Category', 
                 marker='o', linewidth=2.5, markersize=8, errorbar='sd')
    
    plt.title(f"Throughput Degradation Curves (Temp: {temp})", pad=15, fontweight='bold')
    plt.xlabel("Resource Stress Intensity (Number of Workers)")
    plt.ylabel("Generation Throughput (tokens/s)")
    plt.xticks([0, 4]) # Mostriamo solo W=0 (Baseline) e W=4 (Stress)
    plt.legend(title="Resource Class", loc="lower left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] Curva di degradazione salvata: {filename}")

def generate_sensitivity_heatmap(df, temp, filename):
    """Crea la mappa di sensibilità. La pivot calcolerà in automatico la media delle 10 run"""
    df_temp = df[df['Temperatura'] == str(temp)].copy()
    
    # Calcolo media delle 10 baseline
    baseline_mean = df_temp[df_temp['Stress_Category'] == "No Stress"]['Generation_Throughput_ts'].mean()
    if pd.isna(baseline_mean) or baseline_mean == 0:
        return
        
    df_stress = df_temp[df_temp['Stress_Category'] != "No Stress"]
    
    # aggfunc='mean' collassa le 10 ripetizioni in un singolo valore medio per la heatmap
    pivot_df = df_stress.pivot_table(index='Stress_Category', columns='Workers', values='Generation_Throughput_ts', aggfunc='mean')
    
    # Calcoliamo il decremento percentuale vs Media Baseline
    percentage_drop = ((pivot_df - baseline_mean) / baseline_mean) * 100
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentage_drop, annot=True, fmt=".1f", cmap="Reds_r", cbar_kws={'label': '% Throughput Drop vs Baseline'})
    
    plt.title(f"Resource-Sensitivity Map (Temp: {temp})", pad=15, fontweight='bold')
    plt.xlabel("Stressor Intensity (Workers)")
    plt.ylabel("Saturated Resource Class")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] Mappa di sensibilità salvata: {filename}")

def generate_boxplot(df, x_col, y_col, hue_col, title, ylabel, filename, log_scale, order):
    """Genera e salva un boxplot (che ora funzionerà magicamente grazie alle 10 ripetizioni)"""
    plt.figure(figsize=(14, 7))
    palette = "Set2" if hue_col == 'Modello' else "coolwarm"
    
    ax = sns.boxplot(
        data=df, x=x_col, y=y_col, hue=hue_col, 
        order=order, palette=palette, width=0.75, fliersize=4
    )
    
    if log_scale:
        ax.set_yscale('log')
        
    plt.title(title, pad=20, fontsize=15, fontweight='bold')
    plt.xlabel("Stress Scenario", labelpad=12)
    plt.ylabel(ylabel, labelpad=12)
    
    legend_title = "Modelli (Parametri)" if hue_col == 'Modello' else "Temperatura (Temp)"
    plt.legend(title=legend_title, loc="upper left", frameon=True)
    
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato Boxplot: {filename}")

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERRORE] File '{CSV_FILE}' non trovato.")
        return

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    
    if 'Temperatura' not in df.columns:
        print("[ERRORE] Colonna 'Temperatura' non trovata.")
        return

    df['Temperatura'] = df['Temperatura'].astype(str)
    df = df.dropna(subset=['Stressor'])
    
    parsed_meta = df['Stressor'].apply(parse_stress_metadata)
    df['Stress_Category'] = [x[0] for x in parsed_meta]
    df['Workers'] = [x[1] for x in parsed_meta]
    
    # Calcolo Latenze
    df['Prompt_Latency_ms'] = df['Prompt_Throughput_ts'].apply(lambda x: (1000.0 / x) if x > 0 else None)
    df['Generation_Latency_ms'] = df['Generation_Throughput_ts'].apply(lambda x: (1000.0 / x) if x > 0 else None)
    
    # Ordine degli stressor
    stress_order = ["No Stress", "CPU Matrix", "CPU FFT", "MEM Stream", "MEM VM", "CACHE L1", "SCHED Switch"]
    df_filtered = df[df['Stress_Category'].isin(stress_order)].copy()
    
    if df_filtered.empty:
        print("[ERRORE] DataFrame filtrato vuoto. Controlla i nomi degli stressor.")
        return

    metrics_to_plot = {
        "Generation_Throughput_ts": {"name": "Generation Throughput", "ylabel": "Tokens / Sec", "log": False},
        "TTFT_ms": {"name": "Time to First Token (TTFT)", "ylabel": "Latency (ms)", "log": False},
        "Jitter_ms": {"name": "Inter-Token Jitter", "ylabel": "Jitter (ms)", "log": False},
        "Temp_C_Max": {"name": "Peak CPU Temperature", "ylabel": "Peak Temperature (°C)", "log": False},
        "Context_Switches_per_sec": {"name": "OS Context Switches Rate", "ylabel": "Context Switches / sec [Log]", "log": True},
        "Cache_Misses": {"name": "CPU Cache Misses Rate", "ylabel": "Cache Misses / sec [Log]", "log": True},
        "Branch_Misses": {"name": "CPU Branch Misses Rate", "ylabel": "Branch Misses / sec [Log]", "log": True},
        "Power_mW_Avg": {"name": "Average Power Consumption", "ylabel": "Power (mW)", "log": False}
    }
    
    print("\n=== GENERAZIONE GRAFICI (10 RUNS MODE) ===")
    for temp in df_filtered['Temperatura'].unique():
        generate_degradation_curves(df_filtered, temp, f"task6_degradation_curves_temp_{temp}_gpu.png")
        generate_sensitivity_heatmap(df_filtered, temp, f"task6_sensitivity_heatmap_temp_{temp}_gpu.png")

    print("\n=== GENERAZIONE BOXPLOT STATISTICI ===")
    temperatures = df_filtered['Temperatura'].unique()
    for temp in temperatures:
        df_temp = df_filtered[df_filtered['Temperatura'] == temp]
        for col_name, config in metrics_to_plot.items():
            if col_name not in df_temp.columns:
                continue
            title = f"{config['name']} (Fixed Temp: {temp})"
            filename = f"box_dim_{col_name}_temp_{temp}_gpu.png"
            generate_boxplot(df_temp, 'Stress_Category', col_name, 'Modello', title, config['ylabel'], filename, config['log'], stress_order)

    print("\n[SUCCESSO] Tutti i grafici statistici con Branch Misses integrati sono pronti!")

if __name__ == "__main__":
    main()

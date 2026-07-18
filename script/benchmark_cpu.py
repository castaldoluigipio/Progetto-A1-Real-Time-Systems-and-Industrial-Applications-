import subprocess
import time
import csv
import os
import re
import json
import math

# ==================== CONFIGURAZIONE MODELLI E TASK ====================
MODELS_CONFIG = {
    "Qwen2.5-0.5B-Instruct": "models_slm/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen2.5-1.5B-Instruct": "models_slm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Qwen2.5-3B-Instruct": "models_slm/qwen2.5-3b-instruct-q4_k_m.gguf",
    "Qwen2.5-7B-Instruct": "models_slm/qwen2.5-7b-instruct-q4_k_m.gguf"
}

TASKS_CONFIG = {
    "JSON_Extraction": {
        "prompt": "Converti questo testo in JSON:\nnome: Mario, anni: 30\nJSON:",
        "prefix": "{"
    }
    # "Arithmetic": {
    #     "prompt": "Risolvi questa operazione matematica scrivendo SOLO il numero finale, senza ripetere l'operazione e senza aggiungere altro testo:\n432 * 89 =",
    #     "prefix": ""
    # },
    # "MMLU_QA": {
    #     "prompt": "Rispondi alla seguente domanda indicando esclusivamente la lettera della risposta corretta (A, B, C o D) senza aggiungere spiegazioni.\nDomanda: Qual è la capitale della Francia?\nA) Berlino\nB) Madrid\nC) Parigi\nD) Roma\nRisposta:",
    #     "prefix": ""
    # }
}

TEMPERATURES = [0.0, 0.5, 0.7, 1.0]
BIN_PATH = "./build/bin/llama-completion"
RESULTS_FILE = "benchmark_results_cpu.csv"

BASELINE_TEXTS = {}
# =======================================================================

def get_hardware_metrics():
    temp = 0.0
    power_mw = 0.0
    
    # 1. Lettura Temperatura
    thermal_paths = ["/sys/devices/virtual/thermal/thermal_zone0/temp", "/sys/class/thermal/thermal_zone0/temp"]
    for path in thermal_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    temp = float(f.read().strip()) / 1000.0
                    break
            except: pass
            
    # 2. Lettura Dinamica del Consumo della CPU
    try:
        base_hwmon = "/sys/class/hwmon"
        found = False
        for hwmon_dir in os.listdir(base_hwmon):
            dir_path = os.path.join(base_hwmon, hwmon_dir)
            
            for file in os.listdir(dir_path):
                if file.endswith("_label"):
                    try:
                        with open(os.path.join(dir_path, file), "r") as f:
                            label_name = f.read().strip().upper()
                        
                        # Identifichiamo il canale dedicato alla CPU (es. VDD_CPU)
                        if "CPU" in label_name and "GPU" not in label_name:
                            # Estraiamo SOLO il numero del canale dal nome del file (es: da "in2_label" prendiamo "2")
                            channel_num = "".join([c for c in file if c.isdigit()])
                            
                            # STRADA A: Vedi se il kernel espone direttamente il file powerX_input (uW)
                            power_path = os.path.join(dir_path, f"power{channel_num}_input")
                            if os.path.exists(power_path):
                                with open(power_path, "r") as f_p:
                                    power_mw = float(f_p.read().strip()) / 1000.0
                                found = True
                                break
                            
                            # STRADA B: Accoppiata nativa Volt (inX_input) e Corrente (currX_input)
                            v_path = os.path.join(dir_path, f"in{channel_num}_input")
                            i_path = os.path.join(dir_path, f"curr{channel_num}_input")
                            
                            if os.path.exists(v_path) and os.path.exists(i_path):
                                with open(v_path, "r") as f_v, open(i_path, "r") as f_i:
                                    # (Volt_mV * Current_mA) / 1000 = mW
                                    power_mw = (float(f_v.read().strip()) * float(f_i.read().strip())) / 1000.0
                                found = True
                                break
                    except:
                        pass
            if found: break
    except: 
        power_mw = float('nan')
        
    return temp, power_mw if power_mw > 0 else float('nan')

# Frequenza
def get_cpu_frequencies():
    frequencies = []
    try:
        base_path = "/sys/devices/system/cpu"
        for cpu in [d for d in os.listdir(base_path) if re.match(r"^cpu\d+$", d)]:
            freq_file = os.path.join(base_path, cpu, "cpufreq/scaling_cur_freq")
            if os.path.exists(freq_file):
                with open(freq_file, "r") as f:
                    frequencies.append(float(f.read().strip()) / 1000.0)
    except: pass
    return frequencies

# Context Switches
def get_system_context_switches():
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("ctxt"): return int(line.split()[1])
    except: pass
    return 0

# Validità JSON
def evaluate_json_validity(text):
    if not text: return 0.0
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return 1.0
        except json.JSONDecodeError: pass
    return 0.0

# Similarità Lessicale
def calculate_lexical_similarity(text, baseline_text):
    if not text or not baseline_text: return 0.0
    words_text = re.findall(r"\w+", text.lower())
    words_base = re.findall(r"\w+", baseline_text.lower())
    if not words_text or not words_base: return 0.0
    set_text, set_base = set(words_text), set(words_base)
    unigram_similarity = len(set_text.intersection(set_base)) / len(set_text.union(set_base))
    return round(unigram_similarity, 4)

# Salvataggio file.csv
def save_to_csv(model_name, task_name, temperature, stressor_name, prompt_ts, gen_ts, ttft_ms, jitter_ms, 
                avg_temp, max_temp, avg_power, json_valid, lex_sim, avg_cpu_freq, ctxt_rate, cache_misses, branch_misses):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Modello", "Task", "Temperatura", "Stressor", 
                "Prompt_Throughput_ts", "Generation_Throughput_ts", "TTFT_ms", "Jitter_ms",
                "Temp_C_Avg", "Temp_C_Max", "Power_mW_Avg", "JSON_Validity", "Lexical_Similarity",
                "CPU_Freq_MHz_Avg", "Context_Switches_per_sec", "Cache_Misses", "Branch_Misses"
            ])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), model_name, task_name, temperature, stressor_name, 
            prompt_ts, gen_ts, round(ttft_ms, 2), round(jitter_ms, 2), round(avg_temp, 2), round(max_temp, 2), 
            "NaN" if math.isnan(avg_power) else round(avg_power, 2), json_valid, lex_sim, round(avg_cpu_freq, 2), 
            round(ctxt_rate, 2), cache_misses, branch_misses
        ])

# Benchmark di esecuzione
def run_benchmark(model_display_name, model_path, task_name, task_info, temperature, stressor_name, stress_cmd=None):
    print(f"\n[START] Modello: {model_display_name} | Task: {task_name} | Temp: {temperature} | Carico: {stressor_name}")
    stress_process = None
    if stress_cmd:
        stress_process = subprocess.Popen(stress_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    formatted_prompt = (
        f"<|im_start|>system\nSei un assistente di benchmark preciso. Rispondi in modo conciso "
        f"seguendo rigidamente le istruzioni fornite senza aggiungere altro testo o ripetere la domanda.<|im_end|>\n"
        f"<|im_start|>user\n{task_info['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if task_info["prefix"]:
        formatted_prompt += task_info["prefix"]

    cmd = [
        "perf", "stat", "-e", "cache-misses,branch-misses",
        BIN_PATH, "-m", model_path, "-p", formatted_prompt, "-n", "128", 
        "--seed", "42", "-no-cnv", "--temp", str(temperature), "-ngl", "0",
        "-t", "4"
    ]
    
    ctxt_start = get_system_context_switches()
    start_time = time.time()

    llama_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
    
    stdout_lines = []
    temp_samples, power_samples, cpu_freq_samples, token_timestamps = [], [], [], []
    ttft_ms = 0.0
    first_token_captured = False
    
    current_line = ""
    
    # ================= TTFT E JITTER =================
    while True:
        # Leggiamo un singolo carattere alla volta per la massima risoluzione temporale
        char = llama_process.stdout.read(1)
        if not char and llama_process.poll() is not None: break
        
        if char:
            current_time = time.time()
            current_line += char
            
            if char == '\n':
                stdout_lines.append(current_line)
                current_line = ""
                
            # Campionamento hardware continuo ad ogni carattere stampato
            t, p = get_hardware_metrics()
            if t > 0: temp_samples.append(t)
            if not math.isnan(p): power_samples.append(p)
            freqs = get_cpu_frequencies()
            if freqs: cpu_freq_samples.append(sum(freqs) / len(freqs))
            
            # Calcolo dinamico dello stato dell'output
            accumulated_output = "".join(stdout_lines) + current_line
            
            if "assistant" in accumulated_output:
                idx = accumulated_output.find("assistant")
                # Offset: parola 'assistant' + '\n' + eventuale prefisso (es: '{')
                prompt_end_offset = len("assistant") + 1 + len(task_info["prefix"])
                prompt_end_idx = idx + prompt_end_offset
                
                # Se la lunghezza supera il prompt_end_idx, siamo nel VERO primo token generato
                if len(accumulated_output) > prompt_end_idx:
                    if not first_token_captured:
                        # VERO TTFT: tempo dall'avvio fino al primo carattere della risposta reale
                        ttft_ms = (current_time - start_time) * 1000.0
                        first_token_captured = True
                    
                    # Registriamo il timestamp solo per i caratteri generati (VERO Jitter)
                    token_timestamps.append(current_time)
                    
    if current_line:
        stdout_lines.append(current_line)
    # =============================================================================
        
    stdout_remainder, stderr_output = llama_process.communicate()
    if stdout_remainder: stdout_lines.append(stdout_remainder)
        
    elapsed_time = time.time() - start_time
    ctxt_end = get_system_context_switches()

    jitter_ms = 0.0
    if len(token_timestamps) > 2:
        intervals = [(token_timestamps[i] - token_timestamps[i-1]) * 1000.0 for i in range(1, len(token_timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        jitter_ms = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    
    if stress_process:
        stress_process.terminate()
        stress_process.wait()

    avg_temp = sum(temp_samples) / len(temp_samples) if temp_samples else 0.0
    max_temp = max(temp_samples) if temp_samples else 0.0
    avg_power = sum(power_samples) / len(power_samples) if power_samples else float('nan')
    avg_cpu_freq = sum(cpu_freq_samples) / len(cpu_freq_samples) if cpu_freq_samples else 0.0
    ctxt_rate = ((ctxt_end - ctxt_start) / elapsed_time) if elapsed_time > 0 else 0.0

    matches = re.findall(r"eval time.*?([\d.]+)\s+tokens per second", stderr_output)
    prompt_ts = float(matches[0]) if len(matches) >= 1 else 0.0
    gen_ts = float(matches[1]) if len(matches) >= 2 else 0.0

    cm_match = re.search(r'([\d.,]+)\s+cache-misses', stderr_output, re.IGNORECASE)
    bm_match = re.search(r'([\d.,]+)\s+branch-misses', stderr_output, re.IGNORECASE)
    cache_misses = int(cm_match.group(1).replace('.', '').replace(',', '')) if cm_match else 0
    branch_misses = int(bm_match.group(1).replace('.', '').replace(',', '')) if bm_match else 0

    # ================= PARSING OUTPUT =================
    raw_generation = "".join(stdout_lines)
    raw_generation = raw_generation.replace("[end of text]", "").strip()

    if "assistant" in raw_generation:
        generated_text = raw_generation.split("assistant")[-1].strip()
    else:
        clean_prompt = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", formatted_prompt)
        generated_text = raw_generation.replace(clean_prompt, "").strip()
    
    generated_text = re.sub(r"^\s*:\s*", "", generated_text).strip() 

    # Se il testo estratto ha già il prefisso (es: '{'), NON aggiungerlo di nuovo
    if task_info["prefix"] and not generated_text.startswith(task_info["prefix"]):
        full_output_text = task_info["prefix"] + generated_text
    else:
        full_output_text = generated_text
        
    full_output_text = full_output_text.strip()
    # ============================================================================
    
    json_valid = evaluate_json_validity(full_output_text) if task_name == "JSON_Extraction" else 0.0
    
    # ================= SALVATAGGIO E CONFRONTO BASELINE ADATTIVO =================
    if temperature == 0.0 and stressor_name == "Baseline_NoStress":
        # Salviamo la VERA baseline solo alla PRIMA ripetizione in assoluto
        if (model_display_name, task_name) not in BASELINE_TEXTS:
            BASELINE_TEXTS[(model_display_name, task_name)] = full_output_text
            lex_sim = 1.0
            print(f"[GROUND TRUTH ASSOLUTO REGISTRATO] Salvata baseline per {model_display_name} su {task_name}.")
        else:
            # Per le ripetizioni da 2 a 10 a Temp 0, verifichiamo che sia deterministico
            base_text = BASELINE_TEXTS[(model_display_name, task_name)]
            lex_sim = calculate_lexical_similarity(full_output_text, base_text)
    else:
        base_text = BASELINE_TEXTS.get((model_display_name, task_name), "")
        lex_sim = calculate_lexical_similarity(full_output_text, base_text)
    # ============================================================================
        
    print(f"\n[RISULTATI NLP] Validità JSON: {json_valid} | Similarità Lessicale vs Temp 0: {lex_sim * 100:.1f}%")
    print(f"[TESTO EFFETTIVO GENERATO]: {full_output_text}")
    
    save_to_csv(model_display_name, task_name, temperature, stressor_name, prompt_ts, gen_ts, ttft_ms, jitter_ms, 
                avg_temp, max_temp, avg_power, json_valid, lex_sim, avg_cpu_freq, ctxt_rate, cache_misses, branch_misses)


if __name__ == "__main__":
    print("=== INIZIO BENCHMARK GENERALE (10 RIPETIZIONI) ===")
    
    REPETITIONS = 10
    WORKERS_FIXED = 4
    
    STRESSORS_CONFIG = [
        {"name": "CPU_Matrix", "cmd": "/usr/bin/stress-ng --matrix {workers} --timeout 30s"},
        {"name": "CPU_FFT", "cmd": "/usr/bin/stress-ng --fft {workers} --timeout 30s"},
        {"name": "MEM_Stream", "cmd": "/usr/bin/stress-ng --stream {workers} --timeout 30s"},
        {"name": "MEM_VM", "cmd": "/usr/bin/stress-ng --vm {workers} --vm-bytes {vm_bytes}M --timeout 30s"},
        {"name": "CACHE_L1", "cmd": "/usr/bin/stress-ng --l1cache {workers} --timeout 30s"},
        {"name": "SCHED_Switch", "cmd": "/usr/bin/stress-ng --switch {workers} --timeout 30s"}
    ]

    for model_name, model_path in MODELS_CONFIG.items():
        if not os.path.exists(model_path):
            print(f"\n[SKIP] Modello non trovato: '{model_path}'.")
            continue
        
        for temp in TEMPERATURES:
            for t_name, t_info in TASKS_CONFIG.items():
                
                # 1. Esecuzione BASELINE (10 ripetizioni per la temperatura corrente)
                for rep in range(1, REPETITIONS + 1):
                    print(f"\n--- [TEMP {temp}] BASELINE: RIPETIZIONE {rep}/{REPETITIONS} ---")
                    run_benchmark(model_name, model_path, t_name, t_info, temp, "Baseline_NoStress", stress_cmd=None)
                    time.sleep(5)
                
                # 2. Esecuzione STRESSORS (10 ripetizioni per ogni stressor, solo a W=4)
                for stress_cfg in STRESSORS_CONFIG:
                    for rep in range(1, REPETITIONS + 1):
                        print(f"\n--- [TEMP {temp}] STRESS {stress_cfg['name']}: RIPETIZIONE {rep}/{REPETITIONS} ---")
                        
                        vm_bytes_scaled = WORKERS_FIXED * 16 
                        formatted_cmd = stress_cfg["cmd"].format(workers=WORKERS_FIXED, vm_bytes=vm_bytes_scaled)
                        unique_stressor_name = f"{stress_cfg['name']}_W{WORKERS_FIXED}"
                        
                        run_benchmark(model_name, model_path, t_name, t_info, temp, unique_stressor_name, stress_cmd=formatted_cmd)
                        time.sleep(4)
                        
    print("\n=== BENCHMARK COMPLETATO ===")

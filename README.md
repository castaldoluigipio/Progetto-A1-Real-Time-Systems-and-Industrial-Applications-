EdgeAI real-time interference: a performance study
Questa sezione guida passo dopo passo chiunque voglia replicare gli esperimenti direttamente sulla board NVIDIA Jetson.

Una volta effettuato l'accesso alla board (tramite terminale o SSH), la prima cosa da fare è assicurarsi che i tool di stress e compilazione siano installati. Eseguire questi comandi per aggiornare il sistema e installare "stress-ng" e le altre dipendenze necessarie:

sudo apt update && sudo apt upgrade -y
sudo apt install stress-ng build-essential cmake git python3-pip -y

Fatto questo, è opportuno installare i pacchetti Python necessari per eseguire gli script:

pip3 install pandas matplotlib seaborn numpy openai

Dopodiché, direttamente nella cartella llama.cpp (per l'installazione, controllare il file txt nella cartella models), eseguire i codici "benchmark_gpu.py" e "benchmark_cpu.py":

python3 benchmark_gpu.py/benchmark_cpu.py

A valle dei risultati, eseguire i rispettivi codici "plot_results_gpu" e "plot_results_gpu" per plottare i risultati ottenuti. 




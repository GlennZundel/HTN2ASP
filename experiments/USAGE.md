# HTN Planning Experiment auf bwUniCluster - Anleitung

## 📋 Übersicht

Diese Anleitung zeigt, wie du die HTN Planning Experimente auf dem bwUniCluster ausführst, überwachst und Ergebnisse abrufst.

## 🚀 Experiment starten

### 1. Dateien auf Cluster hochladen

**Von deinem lokalen Rechner:**
```bash
scp -r experiments/ username@bwunicluster.scc.kit.edu:/home/username/path/to/
```

Oder mit rsync (empfohlen, überspringt bereits vorhandene Dateien):
```bash
rsync -avz --progress experiments/ username@bwunicluster.scc.kit.edu:/home/username/path/to/experiments/
```

### 2. Auf Cluster einloggen

```bash
ssh username@bwunicluster.scc.kit.edu
```

### 3. In experiments/ Verzeichnis wechseln

```bash
cd /home/username/path/to/experiments/
```

### 4. Experiment ausführen

**⚠️ Dies läuft auf der Login-Node, ist aber sicher! Lab submitted nur Jobs, führt keine schweren Berechnungen aus.**

```bash
python3 experiment.py
```

**Was passiert:**
- Lab erstellt Experiment-Verzeichnis (z.B. `data/`)
- Kopiert Ressourcen in Run-Verzeichnisse
- Generiert SLURM-Job-Scripts
- Submitted 2 Jobs (robotDomain-problem01, factories-simple-problem01)
- Zeigt Job-IDs an:
  ```
  Submitted batch job 12345
  Submitted batch job 12346
  ```

## 📊 Status überwachen

### Jobs in der Queue anzeigen

**Alle deine Jobs:**
```bash
squeue -u $USER
```

**Output-Beispiel:**
```
JOBID  PARTITION  NAME                      USER     ST  TIME  NODES  NODELIST(REASON)
12345  single     robotDomain-problem01     username R   2:34  1      node042
12346  single     factories-simple-probl... username PD  0:00  1      (Priority)
```

**Status-Codes:**
- `PD` = Pending (wartet in Queue)
- `R` = Running (läuft gerade)
- `CG` = Completing (wird beendet)
- Nichts = Fertig

### Detaillierte Job-Info

```bash
scontrol show job 12345
```

### Job-Logs in Echtzeit verfolgen

**Während Job läuft:**
```bash
# Finde das richtige Run-Verzeichnis
ls -lt data/

# Tail das Log
tail -f data/*/runs-*/robotDomain-problem01/run.log
```

**Alternative (wenn Pfad bekannt):**
```bash
tail -f data/exp-2024-01-15-14-30/runs-00001-00002/robotDomain-problem01/run.log
```

### Alle laufenden Logs gleichzeitig

```bash
watch -n 5 'ls -lh data/*/runs-*/*/run.log | tail -20'
```

## 📁 Experiment-Verzeichnisstruktur

Nach `experiment.py` Ausführung:

```
experiments/
├── experiment.py                    # Dein Script
├── bwuni_environment.py             # Environment Config
├── benchmarks/                      # Deine Benchmarks
├── framework/                       # ASP Framework
├── scripts/                         # Pipeline Scripts
└── data/                            # ← Lab erstellt das
    └── exp-2024-01-15-14-30/        # Timestamp
        ├── runs-00001-00002/        # Run-Batch
        │   ├── robotDomain-problem01/
        │   │   ├── domain.hddl
        │   │   ├── problem.hddl
        │   │   ├── framework.lp
        │   │   ├── run_pipeline.py
        │   │   ├── hddl_to_lp.py
        │   │   ├── parseResult.py
        │   │   ├── domain_output.lp      # Generiert
        │   │   ├── problem_output.lp     # Generiert
        │   │   ├── primitives.txt        # Generiert
        │   │   ├── clingo_output.txt     # Generiert
        │   │   ├── orderedtasklist.txt   # ← Lösung!
        │   │   ├── run.log               # Execution log
        │   │   ├── run.err               # Error log
        │   │   └── properties            # Lab properties
        │   └── factories-simple-problem01/
        │       └── ... (gleiche Struktur)
        └── results.html                  # Experiment Report
```

## 🔍 Ergebnisse einsehen

### Geordnete Task-Liste (Hauptergebnis)

**Für robotDomain:**
```bash
cat data/*/runs-*/robotDomain-problem01/orderedtasklist.txt
```

**Beispiel-Output:**
```
move(c, r1, 0)
open(d13, 1)
move(r1, r3, 2)
pickup(o1, r3, 3)
move(r3, r1, 4)
move(r1, r2, 5)
putdown(o1, r2, 6)
```

### Clingo-Output (vollständiger Solver-Output)

```bash
cat data/*/runs-*/robotDomain-problem01/clingo_output.txt
```

**Enthält:**
- Answer Sets
- `taskTBA(...)` Prädikate
- Solver-Statistiken
- Grounding-Info

### Execution Log

```bash
cat data/*/runs-*/robotDomain-problem01/run.log
```

**Enthält:**
- Python Script Output
- Translation-Status
- Clingo-Ausführung
- Parsing-Ergebnisse

### Error Log

```bash
cat data/*/runs-*/robotDomain-problem01/run.err
```

**Nur bei Fehlern:**
- Python Exceptions
- Clingo Errors
- SLURM Warnings

### Lab Properties (Laufzeit, Memory)

```bash
cat data/*/runs-*/robotDomain-problem01/properties
```

**Beispiel:**
```
domain: robotDomain
problem: problem01
time: 45.3
memory: 2341
error:
returncode: 0
```

### HTML Report (Übersicht alle Runs)

**Report herunterladen und lokal öffnen:**
```bash
# Auf deinem lokalen Rechner:
scp username@bwunicluster.scc.kit.edu:/home/username/path/to/experiments/data/*/results.html .
firefox results.html
```

**Oder auf Cluster mit Text-Browser:**
```bash
w3m data/*/results.html
```

## 🛠️ Nützliche Befehle

### Alle Ergebnisse auf einmal anzeigen

```bash
# Alle orderedtasklist.txt Dateien
find data/ -name "orderedtasklist.txt" -exec echo "=== {} ===" \; -exec cat {} \; -exec echo "" \;
```

### Laufzeiten vergleichen

```bash
# Extrahiere 'time' aus allen properties
grep "^time:" data/*/runs-*/*/properties
```

### Memory-Verwendung vergleichen

```bash
# Extrahiere 'memory' aus allen properties
grep "^memory:" data/*/runs-*/*/properties
```

### Fehler finden

```bash
# Suche nach non-zero returncodes
grep "^returncode:" data/*/runs-*/*/properties | grep -v "returncode: 0"
```

### Run-Verzeichnisse nach Größe sortieren

```bash
du -sh data/*/runs-*/*/ | sort -h
```

## 🔄 Experiment wiederholen

**Wenn du Änderungen machst (z.B. andere Benchmarks, andere Limits):**

1. Editiere `experiment.py`
2. Führe erneut aus:
   ```bash
   python3 experiment.py
   ```
3. Lab erstellt neues Experiment-Verzeichnis mit neuem Timestamp

**Alte Experimente bleiben erhalten in `data/`**

## 🧹 Aufräumen

### Experiment-Daten löschen

```bash
# Vorsicht! Löscht alle Ergebnisse
rm -rf data/
```

### Einzelnes Experiment löschen

```bash
rm -rf data/exp-2024-01-15-14-30/
```

### Nur Run-Dateien behalten, Zwischenergebnisse löschen

```bash
# Löscht große intermediate files
find data/ -name "clingo_output.txt" -delete
find data/ -name "domain_output.lp" -delete
find data/ -name "problem_output.lp" -delete
```

## ❌ Troubleshooting

### Job hängt in Queue (PD Status)

**Problem:** Job startet nicht

**Lösung:**
```bash
# Grund anzeigen
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
```

**Häufige Gründe:**
- `Priority` = Warte auf höhere Priorität
- `Resources` = Nicht genug freie Nodes
- `QOSMaxJobsPerUserLimit` = Zu viele Jobs gleichzeitig

### Job schlägt sofort fehl

**Problem:** Job beendet sich mit Fehler

**Schritte:**
1. Error Log prüfen:
   ```bash
   cat data/*/runs-*/RUNNAME/run.err
   ```

2. Execution Log prüfen:
   ```bash
   cat data/*/runs-*/RUNNAME/run.log
   ```

3. Properties prüfen:
   ```bash
   cat data/*/runs-*/RUNNAME/properties
   ```

**Häufige Fehler:**
- `clingo: command not found` → Clingo Modul laden
- `ImportError: No module named 'lab'` → Lab nicht installiert
- `MemoryError` → Memory Limit erhöhen in `experiment.py`

### Leere orderedtasklist.txt

**Problem:** Datei existiert, ist aber leer

**Ursache:** Clingo fand keine Lösung

**Prüfen:**
```bash
cat data/*/runs-*/RUNNAME/clingo_output.txt
```

**Suche nach:**
- `UNSATISFIABLE` = Problem hat keine Lösung
- `TIMEOUT` = Zeit war nicht ausreichend
- `UNKNOWN` = Memory war nicht ausreichend

**Lösungen:**
- Zeit erhöhen: `time_limit=3600` in `experiment.py`
- Memory erhöhen: `memory_limit=16000` in `experiment.py`
- Framework anpassen: mehr Zeitschritte in `framework.lp`

### Python Module fehlen

**Problem:** `ModuleNotFoundError`

**Lösung auf bwUniCluster:**
```bash
# Python3 laden
module load python/3.9

# Lab installieren (in User-Home)
pip3 install --user lab

# Zu experiment.py hinzufügen (am Anfang):
# import sys
# sys.path.insert(0, '/home/username/.local/lib/python3.9/site-packages')
```

### Clingo nicht gefunden

**Problem:** `clingo: command not found`

**Lösung:**

Option 1 - Modul laden (wenn verfügbar):
```bash
module avail clingo
module load clingo
```

Option 2 - Conda Environment:
```bash
module load conda
conda activate potassco  # oder dein Environment
```

Option 3 - In experiment.py Environment-Setup hinzufügen:
```python
env = BWUniEnvironment(
    email="...",
    partition="single",
    setup="module load conda && conda activate potassco"
)
```

## 📧 Email-Benachrichtigungen

Du erhältst Emails an `glenn.zundel@stud.uni-heidelberg.de`:

**Bei Job-Start:** (optional, wenn aktiviert)
- Subject: `SLURM Job_id=12345 Name=robotDomain-problem01 Began`

**Bei Job-Ende:**
- Subject: `SLURM Job_id=12345 Name=robotDomain-problem01 Ended, Run time 00:02:45`
- Enthält: Exit status, Run time, Memory used

**Bei Job-Fehler:**
- Subject: `SLURM Job_id=12345 Name=robotDomain-problem01 Failed`
- Enthält: Error info, Exit code

## 📚 Weiterführende Informationen

### Fast Downward Lab Dokumentation
https://lab.readthedocs.io/

### bwUniCluster Dokumentation
https://wiki.bwhpc.de/e/BwUniCluster_2.0_Slurm_common_Features

### SLURM Befehle
- `squeue` - Jobs anzeigen
- `scancel <jobid>` - Job abbrechen
- `scontrol show job <jobid>` - Job-Details
- `sacct` - Accounting-Info (nach Job-Ende)

## ⚡ Quick Reference

**Experiment starten:**
```bash
cd experiments/
python3 experiment.py
```

**Status checken:**
```bash
squeue -u $USER
```

**Ergebnis anschauen:**
```bash
cat data/*/runs-*/robotDomain-problem01/orderedtasklist.txt
```

**Log verfolgen:**
```bash
tail -f data/*/runs-*/robotDomain-problem01/run.log
```

**Job abbrechen:**
```bash
scancel 12345
```

import re
import sys
import os

def process_tasks(names_file, data_file, result_file):
    # 1. Namen aus names.txt laden
    try:
        with open(names_file, 'r', encoding='utf-8') as f:
            # Set für schnelle Suche, leere Zeilen ignorieren
            valid_names = {line.strip() for line in f if line.strip()}
        print(f"Gelesen: {len(valid_names)} Namen aus '{names_file}'.")
    except FileNotFoundError:
        print(f"Fehler: Datei '{names_file}' nicht gefunden.")
        return

    # 2. Den Raw-String aus output.txt laden
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        print(f"Gelesen: Inhalt aus '{data_file}'.")
    except FileNotFoundError:
        print(f"Fehler: Datei '{data_file}' nicht gefunden.")
        return

    # 3. Parsen und Filtern
    tokens = raw_content.split()
    parsed_items = []
    
    # Regex Muster: taskTBA( INHALT , ZAHL )
    regex_pattern = re.compile(r'taskTBA\((.+?),(\d+)\)')

    for token in tokens:
        match = regex_pattern.match(token)
        
        if match:
            action_body = match.group(1) # z.B. move(locA,locB)
            number_str = match.group(2)  # z.B. 10
            number = int(number_str)

            # Funktionsnamen extrahieren (mit oder ohne Klammer)
            func_name_match = re.match(r'^([a-zA-Z0-9_]+)(\(|$)', action_body)

            if func_name_match:
                func_name = func_name_match.group(1)

                # Nur verarbeiten, wenn der Name in names.txt steht
                if func_name in valid_names:
                    # Unterscheiden: mit oder ohne Argumente
                    if '(' in action_body:
                        # Mit Argumente: move(locA,locB) -> move(locA,locB, 10)
                        formatted_string = f"{action_body[:-1]}, {number})"
                    else:
                        # Ohne Argumente: ok -> ok(10)
                        formatted_string = f"{action_body}({number})"

                    # Tupel speichern: (Sortier-Zahl, Fertiger String)
                    parsed_items.append((number, formatted_string))

    # 4. Sortieren nach der Zahl (erstes Element im Tupel)
    parsed_items.sort(key=lambda x: x[0])

    # 5. In die neue Datei schreiben
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            for _, line in parsed_items:
                f.write(line + '\n')
        print(f"Erfolg: {len(parsed_items)} sortierte Befehle wurden in '{result_file}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}")

if __name__ == "__main__":
    # Example usage:
    # python3 parseResult.py primitives.txt output.txt orderedtasklist.txt

    if len(sys.argv) != 4:
        print("Usage: parseResult.py <primitives> <input_data> <output>")
        print("  <primitives>  - File containing primitive task names (one per line)")
        print("  <input_data>  - File containing clingo output to parse")
        print("  <output>      - File to write sorted task list to")
        sys.exit(1)

    INPUT_NAMES = sys.argv[1]
    INPUT_DATA = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    print("--- Starte Verarbeitung ---")
    process_tasks(INPUT_NAMES, INPUT_DATA, OUTPUT_FILE)
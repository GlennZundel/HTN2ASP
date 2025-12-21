import subprocess
import sys
import os

# Example usage:
# python3 run_pipeline.py domain.hddl problem.hddl framework.lp domain_output.lp problem_output.lp primitives.txt clingo_output.txt orderedtasklist.txt

def run_workflow(domain_file, problem_file, framework_file,
                 domain_output, problem_output, primitives_output,
                 clingo_output, tasklist_output):
    """
    Run the complete HTN planning workflow.

    Args:
        domain_file: Input HDDL domain file
        problem_file: Input HDDL problem file
        framework_file: Input ASP framework file
        domain_output: Output ASP domain file
        problem_output: Output ASP problem file
        primitives_output: Output primitives list file
        clingo_output: Output file for clingo results
        tasklist_output: Output file for ordered task list
    """

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hddl_to_lp_script = os.path.join(script_dir, "hddl_to_lp.py")
    parse_result_script = os.path.join(script_dir, "parseResult.py")

    print("--- 1. Starte Übersetzung (hddl_to_lp) ---")
    try:
        subprocess.run(
            ["python3", hddl_to_lp_script, domain_file, problem_file,
             domain_output, problem_output, primitives_output],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei hddl_to_lp: {e}")
        return False
    except FileNotFoundError:
        print(f"Fehler: Script '{hddl_to_lp_script}' nicht gefunden.")
        return False

    print("\n--- 2. Starte Clingo (Lösen) ---")

    # Check if all required files exist
    clingo_inputs = [problem_output, domain_output, framework_file]
    for f in clingo_inputs:
        if not os.path.exists(f):
            print(f"Fehler: Die Datei '{f}' wurde nicht gefunden.")
            return False

    try:
        with open(clingo_output, "w", encoding="utf-8") as outfile:
            result = subprocess.run(
                ["clingo"] + clingo_inputs,
                stdout=outfile,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Clingo may return non-zero if no model found
            )

        if result.returncode != 0 and result.stderr:
            print(f"Clingo Warnung/Fehler: {result.stderr}")

        print(f"Clingo fertig. Ergebnis in '{clingo_output}' gespeichert.")
    except FileNotFoundError:
        print("Fehler: 'clingo' wurde nicht gefunden. Ist es installiert und im PATH?")
        return False

    print("\n--- 3. Verarbeite Ergebnisse (parseResult) ---")
    try:
        subprocess.run(
            ["python3", parse_result_script, primitives_output,
             clingo_output, tasklist_output],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei parseResult: {e}")
        return False
    except FileNotFoundError:
        print(f"Fehler: Script '{parse_result_script}' nicht gefunden.")
        return False

    print("\n--- Workflow abgeschlossen ---")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: run_pipeline.py <domain> <problem> <framework> <domain_output> <problem_output> <primitives_output> <clingo_output> <tasklist_output>")
        print("\nArguments:")
        print("  <domain>             - Input HDDL domain file")
        print("  <problem>            - Input HDDL problem file")
        print("  <framework>          - Input ASP framework file")
        print("  <domain_output>      - Output ASP domain file")
        print("  <problem_output>     - Output ASP problem file")
        print("  <primitives_output>  - Output primitives list file")
        print("  <clingo_output>      - Output file for clingo results")
        print("  <tasklist_output>    - Output file for ordered task list")
        print("\nExample:")
        print("  python3 run_pipeline.py domain.hddl problem.hddl framework.lp \\")
        print("                          domain_out.lp problem_out.lp primitives.txt \\")
        print("                          clingo_out.txt tasklist.txt")
        sys.exit(1)

    success = run_workflow(
        domain_file=sys.argv[1],
        problem_file=sys.argv[2],
        framework_file=sys.argv[3],
        domain_output=sys.argv[4],
        problem_output=sys.argv[5],
        primitives_output=sys.argv[6],
        clingo_output=sys.argv[7],
        tasklist_output=sys.argv[8]
    )

    sys.exit(0 if success else 1)

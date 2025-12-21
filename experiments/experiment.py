#!/usr/bin/env python3
"""
HTN Planning Experiment for bwUniCluster

This script sets up and executes a Fast Downward Lab experiment for running
HTN planning benchmarks through the HDDL → ASP → Clingo → Result parsing pipeline.

The experiment runs on the bwUniCluster SLURM system with the following flow:
1. HDDL domain and problem files are translated to ASP
2. Clingo solves the ASP encoding
3. Results are parsed into an ordered task list

Each run executes the complete pipeline via run_pipeline.py with appropriate
resource limits (1800s time limit, 8000MB memory for clingo).
"""

import sys
from pathlib import Path
from lab.experiment import Experiment
from lab.reports import Attribute, Report

from bwuni_environment import BWUniEnvironment

# ============================================================================
# Path Configuration
# ============================================================================

EXPERIMENT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = EXPERIMENT_DIR / "benchmarks"
FRAMEWORK_DIR = EXPERIMENT_DIR / "framework"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"

# ============================================================================
# Environment Configuration
# ============================================================================

env = BWUniEnvironment(
    email="glenn.zundel@stud.uni-heidelberg.de",
    memory_per_cpu="8100M",
)

# ============================================================================
# Experiment Setup
# ============================================================================

# Create the experiment with environment
exp = Experiment(environment=env)

# Define benchmark suite: (domain_name, problem_file)
SUITE = [
    ("depots", "problem01.hddl"),
    ("robotDomain", "problem02.hddl"),
    ]

# ============================================================================
# Run Configuration
# ============================================================================

for domain, problem in SUITE:
    # Generate run identifier
    problem_name = problem.replace(".hddl", "")
    run_id = f"{domain}-{problem_name}"

    # Create run
    run = exp.add_run()
    run.set_property("id", [domain, problem_name])
    run.set_property("domain", domain)
    run.set_property("problem", problem_name)

    # Add benchmark resources
    run.add_resource(
        "domain",
        BENCHMARKS_DIR / domain / "domain.hddl",
        "domain.hddl"
    )
    run.add_resource(
        "problem",
        BENCHMARKS_DIR / domain / problem,
        "problem.hddl"
    )

    # Add framework resource
    run.add_resource(
        "framework",
        FRAMEWORK_DIR / "framework.lp",
        "framework.lp"
    )

    # Add script resources
    run.add_resource(
        "run_pipeline",
        SCRIPTS_DIR / "run_pipeline.py",
        "run_pipeline.py"
    )
    run.add_resource(
        "hddl_to_lp",
        SCRIPTS_DIR / "hddl_to_lp.py",
        "hddl_to_lp.py"
    )
    run.add_resource(
        "parseResult",
        SCRIPTS_DIR / "parseResult.py",
        "parseResult.py"
    )

    # Add pipeline command
    # The run_pipeline.py script orchestrates:
    # 1. hddl_to_lp.py: HDDL → ASP translation
    # 2. clingo: ASP solving
    # 3. parseResult.py: Result extraction
    run.add_command(
        "run-pipeline",
        [
            sys.executable, "{run_pipeline}",
            "domain.hddl",           # arg1: domain input
            "problem.hddl",          # arg2: problem input
            "framework.lp",          # arg3: framework input
            "domain_output.lp",      # arg4: domain ASP output
            "problem_output.lp",     # arg5: problem ASP output
            "primitives.txt",        # arg6: primitives list
            "clingo_output.txt",     # arg7: clingo raw output
            "orderedtasklist.txt"    # arg8: final task list
        ],
        time_limit=3600,             # 30 minutes
        memory_limit=8100            # 8GB for clingo
    )

# ============================================================================
# Reporting Configuration
# ============================================================================

# Define attributes to track and report
attributes = [
    Attribute("domain", absolute=False),
    Attribute("problem", absolute=False),
    Attribute("time", absolute=False, min_wins=True),
    Attribute("memory", absolute=False, min_wins=True),
]

# Create HTML report
report = Report(attributes=attributes, format="html")
exp.add_report(report, outfile="results.html")

# ============================================================================
# Execution Steps
# ============================================================================

# Add execution steps for SLURM workflow:
# 1. build  - Create run directories and copy resources
# 2. start  - Submit SLURM jobs
# 3. fetch  - Fetch results after jobs complete (optional, for remote clusters)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)

# Parse command line arguments and execute requested steps
# Usage:
#   python3 experiment.py build       # Build the experiment
#   python3 experiment.py start       # Submit SLURM jobs
#   python3 experiment.py results.html # Generate report (after jobs finish)
#   python3 experiment.py --all       # Run all steps sequentially
exp.run_steps()

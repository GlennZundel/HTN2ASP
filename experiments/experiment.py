#!/usr/bin/env python3
import sys
from pathlib import Path
from lab.experiment import Experiment
from lab.reports import Attribute, Report
from lab.environments import LocalEnvironment

exp_dir = Path(__file__).resolve().parent
benchmarks = exp_dir / "benchmarks"
framework = exp_dir / "framework"
scripts = exp_dir / "scripts"

env = LocalEnvironment(processes=1)
exp = Experiment(environment=env)

SUITE = [('assemblyhierarchical', 'problem01'),
        ('assemblyhierarchical', 'problem02'),
        ('assemblyhierarchical', 'problem03'),
        ('assemblyhierarchical', 'problem04'),
        ('assemblyhierarchical', 'problem05'),
        ('assemblyhierarchical', 'problem06'),
        ('assemblyhierarchical', 'problem07'),
        ('assemblyhierarchical', 'problem08'),
        ('assemblyhierarchical', 'problem09'),
        ('assemblyhierarchical', 'problem10'),
         ('barman-bdi', 'problem01'),
        ('barman-bdi', 'problem02'),
        ('barman-bdi', 'problem03'),
        ('barman-bdi', 'problem04'),
        ('barman-bdi', 'problem05'),
        ('barman-bdi', 'problem06'),
        ('barman-bdi', 'problem07'),
        ('barman-bdi', 'problem08'),
        ('barman-bdi', 'problem09'),
        ('barman-bdi', 'problem10'),
('blocksworld-gtohp', 'problem01'),
        ('blocksworld-gtohp', 'problem02'),
        ('blocksworld-gtohp', 'problem03'),
        ('blocksworld-gtohp', 'problem04'),
        ('blocksworld-gtohp', 'problem05'),
        ('blocksworld-gtohp', 'problem06'),
        ('blocksworld-gtohp', 'problem07'),
        ('blocksworld-gtohp', 'problem08'),
        ('blocksworld-gtohp', 'problem09'),
        ('blocksworld-gtohp', 'problem10'),
('depots', 'problem01'),
        ('depots', 'problem02'),
        ('depots', 'problem03'),
        ('depots', 'problem04'),
        ('depots', 'problem05'),
        ('depots', 'problem06'),
        ('depots', 'problem07'),
        ('depots', 'problem08'),
        ('depots', 'problem09'),
        ('depots', 'problem10'),
('factories-simple', 'problem01'),
        ('factories-simple', 'problem02'),
        ('factories-simple', 'problem03'),
        ('factories-simple', 'problem04'),
        ('factories-simple', 'problem05'),
        ('factories-simple', 'problem06'),
        ('factories-simple', 'problem07'),
        ('factories-simple', 'problem08'),
        ('factories-simple', 'problem09'),
        ('factories-simple', 'problem10'),
('hiking', 'problem01'),
        ('hiking', 'problem02'),
        ('hiking', 'problem03'),
        ('hiking', 'problem04'),
        ('hiking', 'problem05'),
        ('hiking', 'problem06'),
        ('hiking', 'problem07'),
        ('hiking', 'problem08'),
        ('hiking', 'problem09'),
        ('hiking', 'problem10'),
('robot', 'problem01'),
        ('robot', 'problem02'),
        ('robot', 'problem03'),
        ('robot', 'problem04'),
        ('robot', 'problem05'),
        ('robot', 'problem02_01'),
        ('robot', 'problem03_01'),
        ('robot', 'problem03_02'),
        ('robot', 'problem03_03'),
        ('robot', 'problem04_03'),
        ('robot', 'problem05_05'),
('rover-gtohp', 'problem01'),
        ('rover-gtohp', 'problem02'),
        ('rover-gtohp', 'problem03'),
        ('rover-gtohp', 'problem04'),
        ('rover-gtohp', 'problem05'),
        ('rover-gtohp', 'problem06'),
        ('rover-gtohp', 'problem07'),
        ('rover-gtohp', 'problem08'),
        ('rover-gtohp', 'problem09'),
        ('rover-gtohp', 'problem10'),
('satellite', 'problem01'),
        ('satellite', 'problem02'),
        ('satellite', 'problem03'),
        ('satellite', 'problem04'),
        ('satellite', 'problem05'),
        ('satellite', 'problem06'),
        ('satellite', 'problem07'),
        ('satellite', 'problem08'),
        ('satellite', 'problem09'),
        ('satellite', 'problem10'),
('towers', 'problem01'),
        ('towers', 'problem02'),
        ('towers', 'problem03'),
        ('towers', 'problem04'),
        ('towers', 'problem05'),
        ('towers', 'problem06'),
        ('towers', 'problem07'),
        ('towers', 'problem08'),
        ('towers', 'problem09'),
        ('towers', 'problem10'),
         ]

for domain, problem in SUITE:
    prob_name = problem.replace(".hddl", "")
    run = exp.add_run()
    run.set_property("id", [domain, prob_name])
    run.set_property("domain", domain)
    run.set_property('problem', prob_name)
    run.add_resource("domain", benchmarks / domain / "domain.hddl", "domain.hddl")
    run.add_resource("problem", benchmarks / domain / problem, "problem.hddl")
    run.add_resource("framework", framework / "framework.lp", "framework.lp")
    run.add_resource("run_pipeline", scripts / "run_pipeline.py", "run_pipeline.py")
    run.add_resource("hddl_to_lp", scripts / "hddl_to_lp.py", "hddl_to_lp.py")
    run.add_resource("parseResult", scripts / "parseResult.py", "parseResult.py")
    run.add_command(
        "run-pipeline",
        [sys.executable, "{run_pipeline}", "domain.hddl", "problem.hddl", "framework.lp",
         "domain_output.lp", "problem_output.lp", "primitives.txt", "clingo_output.txt", "orderedtasklist.txt"],
        time_limit=1800, memory_limit=8100
    )

attributes = [
    Attribute("domain", absolute=False),
    Attribute("problem", absolute=False),
    Attribute("time", absolute=False, min_wins=True),
    Attribute('memory', absolute=False, min_wins=True),
]
report = Report(attributes=attributes, format="html")
exp.add_report(report, outfile="results.html")
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.run_steps()

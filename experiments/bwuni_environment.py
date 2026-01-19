"""
BWUniCluster environment for Fast Downward Lab experiments.

This module defines a custom SLURM environment optimized for the bwUniCluster
with appropriate memory settings and email notifications for HTN planning experiments.
"""

from lab.environments import BaselSlurmEnvironment


class BWUniEnvironment(BaselSlurmEnvironment):
    """
    Environment for bwUniCluster with optimized memory and email settings.

    Configuration:
    - CPUs per task: 1 (one core per run)
    - Memory per CPU: 8000M (8GB)
    - Time limit: 30 minutes per task
    - Email notifications on job completion/failure
    - Default partition: cpu_il
    """

    def __init__(self, email="glenn.zundel@stud.uni-heidelberg.de",
                 partition="cpu_il", **kwargs):
        """
        Initialize BWUniCluster environment.

        Args:
            email: Email address for SLURM notifications
            partition: SLURM partition to use (default: "cpu_il")
            **kwargs: Additional arguments passed to BaselSlurmEnvironment
        """
        # Configure SLURM options: 1 CPU, 8GB RAM, 30 min timeout
        extra_options = (
            f"#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user={email}\n"
            f"#SBATCH --time=02:00:00\n"
            f"#SBATCH --cpus-per-task=1"
        )

        # Initialize base environment with cluster-specific settings
        super().__init__(
            email=email,
            partition=partition,
            extra_options=extra_options,
            time_limit_per_task="2:00:00",
            cpus_per_task=1,
            **kwargs
        )

    def _get_job_header(self, step, is_last):
        """Override to remove QoS line which is not valid on bwUniCluster."""
        header = super()._get_job_header(step, is_last)
        # Remove the QoS line completely - bwUniCluster does not use QoS
        lines = header.split('\n')
        lines = [line for line in lines if not line.startswith('#SBATCH --qos=')]
        return '\n'.join(lines)

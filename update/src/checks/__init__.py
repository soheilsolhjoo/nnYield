"""
Unified Sanity Check Module Entry Point.

This module aggregates the specialized checking modules (Physics, Diagnostics, 
Reporting) into a single SanityChecker class. This allows the CLI to execute 
the entire validation suite with a single command.
"""

from .core import BaseChecker
from .physics import PhysicsChecks
from .diagnostics import DiagnosticsChecks
from .reporting import ReportingChecks

class SanityChecker(PhysicsChecks, DiagnosticsChecks, ReportingChecks):
    """
    TRICK: THE UNIFIED EXECUTIVE.
    
    By inheriting from multiple specialized check classes, this class acts 
    as a centralized 'Orchestrator'. It has access to all diagnostic methods 
    while keeping the code organized into logical modules.
    """
    
    def run_all(self):
        """
        Executes the full suite of validations in a logical order.
        Matches the functional depth of the reference system.
        """
        print("\n--- Starting Comprehensive Sanity Check ---")
        
        # 1. DIAGNOSTICS (Machine Learning Health)
        # Visualizes the training loss history breakdown
        self.check_loss_history_detailed()
        # Analyzes late-stage convergence stability
        self.check_loss_stability()
        # Maps the individual gradient components across the domain
        self._plot_gradient_components()
        # Single-point dry run of the normality math
        self.check_r_calculation_logic() 
        
        # 2. PHYSICS (Material Behavior)
        # Verifies the mathematical factory consistency
        self.check_benchmark_derivatives()
        # Plots 2D cross-sections of the yield loci
        self.check_2d_loci_slices()
        # Linearized yield radius vs direction (Generates: radius_vs_theta.png)
        self.check_radius_vs_theta()
        # Lankford coefficients and uniaxial stress traces
        self.check_r_values()
        
        # 3. ADVANCED PHYSICS (Thermodynamics & Stability Maps)
        # Global error and curvature heatmaps (Generates: se_error_map.png, convexity_map.png)
        self.check_full_domain_benchmark() 
        # Detailed statistical breakdown of curvature metrics
        self.check_convexity_detailed()
        # 1D equator stability analysis (Generates: convexity_slice_1d.png)
        self.check_convexity_slice_1d()
        
        # 4. REPORTING (Numerical Statistics)
        # Calculates MAE over 2000 points and generates error histograms
        self.check_global_statistics()
        
        print(f"\n✅ All checks complete. Diagnostic plots saved in: {self.plot_dir}")

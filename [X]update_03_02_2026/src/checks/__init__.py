# Import the specialized checking modules
# Each module handles a specific aspect of validation:
# - core: Shared infrastructure (math, sampling, loading)
# - physics: Material science validation (Yield Loci, R-values, Convexity)
# - diagnostics: ML health checks (Loss history, Gradient noise)
# - reporting: Quantitative statistics (MAE, Error distributions)
from .core import BaseChecker
from .physics import PhysicsChecks
from .diagnostics import DiagnosticsChecks
from .reporting import ReportingChecks

class SanityChecker(BaseChecker, PhysicsChecks, DiagnosticsChecks, ReportingChecks):
    """
    The Unified Sanity Checker.
    
    This class combines the functionality of all specific checker modules into a single 
    executable suite. It uses Python's multiple inheritance to aggregate all the 
    validation methods (check_*) into one class.
    
    Usage:
        checker = SanityChecker(model, config, output_dir)
        checker.run_all()
    """
    
    def run_all(self):
        """
        Executes the full suite of validations in a logical order.
        
        The sequence is designed to go from low-level debugging to high-level physics:
        1. **Diagnostics**: Is the model broken? (Losses, simple math checks)
        2. **Physics**: Does it behave like a material? (Yield surface shape, anisotropy)
        3. **Advanced Physics**: Is it thermodynamically valid? (Convexity, stability maps)
        4. **Reporting**: How accurate is it overall? (Global statistics)
        """
        print("--- Starting Sanity Checks ---")
        
        # ---------------------------------------------------------
        # 1. DIAGNOSTICS (Machine Learning Health)
        # ---------------------------------------------------------
        # Visualizes the training loss history (Stress vs R-value vs Convexity)
        self.check_loss_history_detailed()
        
        # Analyzes the tail end of training to check for noise/instability
        self.check_loss_stability()
        
        # Visualizes the smoothness of gradient components (normals)
        self._plot_gradient_components()
        
        # A single-point dry run to ensure R-value math doesn't crash
        self.check_r_calculation_logic() 
        
        # ---------------------------------------------------------
        # 2. PHYSICS (Material Behavior)
        # ---------------------------------------------------------
        # Plots 2D cross-sections of the yield surface (contours)
        self.check_2d_loci_slices()
        
        # Plots yield radius vs angle for plane stress
        self.check_radius_vs_theta()
        
        # Plots R-value anisotropy and Yield Stress vs Angle
        self.check_r_values()
        
        # ---------------------------------------------------------
        # 3. ADVANCED PHYSICS (Thermodynamics & Maps)
        # ---------------------------------------------------------
        # Generates heatmaps of error and gradient deviation across the full domain
        self.check_full_domain_benchmark() 
        
        # Statistical analysis of convexity (Eigenvalues vs Minors)
        self.check_convexity_detailed()
        
        # 1D slice of stability along the equator (most critical region)
        self.check_convexity_slice_1d()
        
        # ---------------------------------------------------------
        # 4. REPORTING (Final Statistics)
        # ---------------------------------------------------------
        # Calculates global MAE for Stress and R-values (Dense sampling)
        # and generates error histograms.
        self.check_global_statistics()
        
        print(f"Done. Plots in '{self.plot_dir}'")
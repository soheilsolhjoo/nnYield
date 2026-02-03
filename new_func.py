    def check_2d_loci_slices(self):
        """
        Plots 2D cross-sections of the yield surface at increasing shear stress levels.
        
        Physics:
        - At Shear=0 (S12=0), we see the standard Plane Stress ellipse.
        - As Shear increases, the elastic domain should shrink (Von Mises / Hill criterion).
        - At Max Shear, the domain should collapse to a point or small region.
        
        Output: plots/yield_loci_slices.png
        """
        print("Running 2D Loci Slices Check...")
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        N = phys.get('N', 1.5)
        max_shear = ref_stress / np.sqrt(2*N)
        
        shear_ratios = [0.0, 0.4, 0.8, 0.95]
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_ratios)))
        theta = np.linspace(0, 2*np.pi, 360).astype(np.float32)
        
        plt.figure(figsize=(7, 7))
        plt.plot([], [], 'k:', linewidth=1.5, label='Benchmark') # Dummy for legend
        
        max_val_plotted = 0.0

        for i, ratio in enumerate(shear_ratios):
            current_s12_val = ratio * max_shear
            
            # 1. Generate Raw Stress Vectors
            # These vectors define the direction in the S11-S22 plane for a fixed S12.
            # DO NOT normalize these; the model's homogeneity handles scaling.
            s11_in = np.cos(theta)
            s22_in = np.sin(theta)
            s12_in = np.full_like(theta, current_s12_val)
            
            raw_inputs = np.stack([s11_in, s22_in, s12_in], axis=1).astype(np.float32)

            # 2. Get NN Predictions
            # The model internally calculates features from these raw stress values.
            pred_se = self.model(tf.constant(raw_inputs)).numpy().flatten()
            rad_nn = ref_stress / (pred_se + 1e-8)

            # 3. Get Benchmark Predictions
            # We pass the same raw inputs to the benchmark for a direct comparison.
            _, (val_vm, _) = self._get_predictions(s11_in, s22_in, s12_in)
            rad_vm = ref_stress / (val_vm + 1e-8)

            # 4. Plot the Surface Points
            # The final plot coordinates are the radius scaled by the original S11/S22 direction.
            s11_plot_nn = rad_nn * s11_in
            s22_plot_nn = rad_nn * s22_in
            
            s11_plot_vm = rad_vm * s11_in
            s22_plot_vm = rad_vm * s22_in

            # Track axis limits
            current_max = max(np.max(np.abs(s11_plot_nn)), np.max(np.abs(s11_plot_vm)))
            if current_max > max_val_plotted: max_val_plotted = current_max

            plt.plot(s11_plot_nn, s22_plot_nn, color=colors[i], linewidth=2, label=f"Shear={ratio:.2f}")

    def check_2d_loci_slices(self):
        print("Running 2D Loci Slices Check...")
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        N = phys.get('N', 1.5)
        max_shear = ref_stress / np.sqrt(2*N)
        
        shear_ratios = [0.0, 0.4, 0.8, 0.95]
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_ratios)))
        theta = np.linspace(0, 2*np.pi, 360)
        
        plt.figure(figsize=(7, 7))
        
        # Plot Benchmark Dummy first for Legend Order
        plt.plot([], [], 'k:', linewidth=1.5, label='Benchmark')

        max_val_plotted = 0.0

        for i, ratio in enumerate(shear_ratios):
            current_s12_val = ratio * max_shear
            s11_in = np.cos(theta); s22_in = np.sin(theta); s12_in = np.full_like(theta, current_s12_val)
            
            (val_nn, _, _), (val_vm, _) = self._get_predictions(s11_in, s22_in, s12_in)
            rad_nn = ref_stress / (val_nn + 1e-8)
            rad_vm = ref_stress / (val_vm + 1e-8)
            
            # Track max value for tighter axis limits
            current_max = max(rad_nn.max(), rad_vm.max())
            if current_max > max_val_plotted: max_val_plotted = current_max
            
            plt.plot(rad_nn*s11_in, rad_nn*s22_in, color=colors[i], linewidth=2, label=f"Shear={ratio:.2f}")
            plt.plot(rad_vm*s11_in, rad_vm*s22_in, color='k', linestyle=':', linewidth=1.5, alpha=0.6)

        # Plot Max Shear Marker
        plt.scatter([0], [0], color='red', marker='x', s=100, label=f'Max Shear ({max_shear:.2f})', zorder=10)
        
        # Tighter Axis Limits
        limit = max_val_plotted * 1.1 # 10% margin
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        
        plt.axis('equal'); plt.xlabel("S11"); plt.ylabel("S22")
        plt.grid(True, alpha=0.3)
        plt.title(f"Yield Loci Slices (Ref={ref_stress})")

        # Legend inside, letting matplotlib find the best empty spot
        plt.legend(loc='best', fontsize='small', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "yield_loci_slices.png")); plt.close()

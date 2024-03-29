## pair_style local/density

This LAMMPS pair style implements the Local Density (LD) potential:- a mean-field manybody potential useful for reproducing manybody effects in a computationally efficient manner in coarse-grained (CG) molecular dynamics (MD) simulations. In the past it has been succesfully applied in constructing CG models of solvent-free polymer folding and liquid-liquid phase behavior. Here, we present its implementation as a LAMMPS pair style, drawing heavily from the pair_eam styles.

For further information, please read the manual. (manual/ld_manual.pdf)

The examples section contains two examples: a) 26.5% (benzene mole fraction) CG benzene-water system where benzene and water are both coarse-grained into single beads and, b) CG methanol in implicit water. Each example contains a data file, tabulated pair potential and LD potential files, and a LAMMPS input script.

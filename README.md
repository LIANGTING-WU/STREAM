# STREAM: Simulation Toolkit for time-Resolved Electronic transport and gas Adsorption Modeling

![STREAM Logo](STREAM-LOGO.png)

STREAM is a Python-based graphical user interface (GUI) application designed to simulate the performance of chemiresistive gas sensors. It combines kinetic modeling of gas adsorption/desorption with electronic transport data from sources like Non-Equilibrium Green's Function (NEGF) calculations to provide a comprehensive analysis toolkit for researchers and engineers in the field of gas sensing.

## Features

- **Interactive GUI**: An easy-to-use interface built with Tkinter for setting parameters, running simulations, and visualizing results in real-time.
- **Kinetic Modeling**:
    - Simulates single and multi-component gas adsorption and desorption dynamics over time.
    - Calculates surface coverage based on user-defined parameters (pressure, temperature, molecule properties).
    - Supports both mobile and immobile transition state assumptions for adsorption kinetics.
- **Electronic Properties Analysis**:
    - Imports and analyzes I-V data from NEGF simulation results (`.csv`).
    - Automatically calculates and plots I-V curves, R-V curves, and sensitivity.
    - Identifies optimal operating voltages by finding peaks in the sensitivity curves.
- **Integrated Sensor Performance Simulation**:
    - Simulates the total resistance of the sensor over time by combining multi-component coverage kinetics with the resistance data of each species.
- **Advanced Analysis**:
    - **Temperature Dependence**: Analyzes how recovery time, equilibrium constant, and equilibrium coverage change with temperature.
    - **Selectivity Analysis**: Generates a selectivity matrix and bar charts to compare the sensor's response to different target gases versus interfering species.
- **Data and Plot Export**:
    - Export all generated plots as high-quality PNG images.
    - Save simulation results (e.g., coverage vs. time, resistance vs. time) to `.dat` and `.txt` files for further analysis.

## Prerequisites

Before running STREAM, ensure you have Python 3 and the following libraries installed:

- NumPy
- Pandas
- SciPy
- Matplotlib

You can install these dependencies using pip:
```bash
pip install numpy pandas scipy matplotlib
```
*Note: Tkinter is usually included with standard Python installations.*

## How to Run

1.  Clone or download the repository.
2.  Navigate to the project directory.
3.  Run the application from your terminal:
    ```bash
    python STREAM.py
    ```

## Using the Application

The GUI is organized into three main sections: **Global Parameters**, **Molecule Parameters**, and the **Plotting Area**.

1.  **Load Molecule Data**:
    - Begin by loading a `molecule_parameters.csv` file using the "Load CSV" button. This file contains the physical parameters for the gas molecules you want to simulate.
2.  **Set Global Parameters**:
    - Adjust the simulation parameters in the "Global Parameters" section, such as pressure, temperature, and simulation time spans.
3.  **Generate Plots**:
    - Select one of the tabs in the plotting area (e.g., "Coverage vs. Time", "Temperature Dependence").
    - Click the **"Generate / Update Plot"** button to run the simulation and display the results.
4.  **Analyze Electronic Properties**:
    - To use the "NEGF Analyzer", "Selectivity", or "Resistance vs. Time" tabs, you must first load a CSV file containing NEGF I-V data.
    - Go to the **"NEGF Analyzer"** tab, click "Generate / Update Plot", and you will be prompted to select your NEGF results file.
    - The tool will automatically process the data, store the results, and suggest an optimal operating voltage. This voltage can then be used for selectivity and resistance simulations.
5.  **Export Results**:
    - Use the "Export Current Plot" button to save the currently displayed figure.
    - Data files (`.dat`, `.txt`) are automatically saved in the application's directory after certain simulations are run.

## Input File Formats

### 1. Molecule Parameters (`molecule_parameters.csv`)

This file contains the physical parameters for each gas molecule. The application uses these values for kinetic calculations.

**Columns:**
- `name`: Name of the molecule (e.g., NH3, CO).
- `p`: Partial pressure of this gas for multi-component simulations (in bar).
- `m`: Molar mass (in amu).
- `Edes`: Desorption energy (in eV).
- `sigma`: Symmetry number of the molecule.
- `molecule_type`: `linear` or `nonlinear`.
- `theta_rot_linear`: Rotational temperature for linear molecules (in K).
- `theta_rot_A`, `theta_rot_B`, `theta_rot_C`: Rotational temperatures for nonlinear molecules (in K).

**Example:**
```csv
name,p,m,Edes,sigma,molecule_type,theta_rot_linear,theta_rot_A,theta_rot_B,theta_rot_C
H2,1,2.016,0.8,2,linear,87.6,0,0,0
NH3,1,17.031,1.2,3,nonlinear,0,13.6,13.6,9.07
```

### 2. NEGF Results (`NEGF_result.csv`)

This file contains the current-voltage (I-V) data from your electronic structure calculations.

**Format:**
- The first column must be **Voltage (V)**.
- The second column must be the current for the **pristine (unfunctionalized) material**.
- Subsequent columns should be the current for the material with a specific gas molecule adsorbed. The column header will be used as the species name.

**Example:**
```csv
Voltage,Pristine,NH3,CO
0.1,0.05,0.04,0.045
0.2,0.10,0.08,0.09
...
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

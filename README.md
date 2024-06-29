Beam Profile Analysis Tool
This repository contains a GUI-based Beam Profile Analysis Tool developed using Python's tkinter library. The tool allows users to load beam profile data, perform various calculations, and visualize the results.

Features
Load beam profile data from a text file.
Calculate important beam profile parameters such as Right IP, Left IP, Average RDV, Field Size, Right Penumbra, and Left Penumbra.
Determine horizontal distances at specific dose levels (90%, 75%, and 60%).
Display results in a text box within the GUI.
Visualize the beam profile using matplotlib.
Requirements
Python 3.x
Required Python packages:
tkinter
pandas
numpy
matplotlib
fpdf
You can install the required packages using pip:

bash
Copy code
pip install pandas numpy matplotlib fpdf
How to Use
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/beam-profile-analysis-tool.git
cd beam-profile-analysis-tool
Run the application:
bash
Copy code
python main.py
Using the GUI:
Click on the "Load Data" button to load a text file containing the beam profile data.
Click on the "Calculate Results" button to perform the calculations and display the results.
The results will be shown in the text box, and a plot of the beam profile will be displayed within the GUI.
Code Overview
main.py: Contains the main application code including the GUI layout, data loading, calculations, and plotting.
Main Functions
load_data(): Opens a file dialog to load the beam profile data from a text file.
calculate_results(): Performs the calculations and updates the GUI with the results and plot.
Results Calculated
Right IP (Rt_RDV)
Left IP (Lt_RDV)
Average RDV (Relative Dose Value)
Field Size (in cm)
Right Penumbra (in mm)
Left Penumbra (in mm)
Horizontal distances at 90%, 75%, and 60% dose levels
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or suggestions, please open an issue in the repository or contact the repository owner.

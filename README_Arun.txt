Data Files:
1) 3Sc_elem_pos.mat - Mat file to store element coordinates of the probe elements
2) ph_specs.mat - Mat file to store specs of the acquisition process and probe
3) sample_data - Actual acquired data
Script Files:
1) dynamicFocusing.py - The original file from the original authors - off Github - it doesn’t even run because of the TF update
1-1) dynamicFocusing_v1.py - The original file modified to run on my machine using TF v1 - **Implemented Scan Conversion at this step**
1-2) dynamicFocusing_v2.py - The original file modified to run on my machine using TF v2
2) dynamicFocusing_PyTorch.py - **The original file modified to run on my machine using PyTorch**
2-1) dynamicFocusing_PyTorch_SinglePWSim.py (and associated folder) - Modified the base PyTorch code to work with simulated single plane wave data.
2-2) dynamicFocusing_PyTorch_SinglePWPhantom.py (and associated folder) - Modified dynamicFocusing_PyTorch_SinglePWSim.py to work with single plane wave phantom data.
2-3) dynamicFocusing_PyTorch_FI.py (and associated folder) - Modified dynamicFocusing_PyTorch_SinglePWPhantom.py to work for Focused Imaging Acquisitons - Tested it with an in-vivo use case, works pretty well

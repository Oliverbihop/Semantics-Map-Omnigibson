<h1 align="center">Construct Semantic Map on Omnigibson Simulation and Navigation</h1>


<p align="center">
  <img src="https://raw.githubusercontent.com/Oliverbihop/Semantics-Map-Omnigibson/main/assets/Screenshot 2025-10-15 162153.png" alt="Robot" width="800"/>
  <img src="https://raw.githubusercontent.com/Oliverbihop/Semantics-Map-Omnigibson/main/assets/map_capture.png" alt="Project Banner" width="800"/>
</p>



<p align="center">
  A semantic mapping framework built on top of Omnigibson and fast_gicp.
</p>

---

## 🚀 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone --recursive https://github.com/Oliverbihop/Semantics-Map-Omnigibson.git
cd Semantics-Map-Omnigibson 
```
### 2️⃣ Install GICP (Python bindings only)
```bash
cd submodules/fast_gicp

# Remove any previous build
rm -rf build

# Install Python bindings
python3 setup.py install --user

cd ../..
```
### 3️⃣ Install Omnigibson (BEHAVIOR-1K)
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Full installation with environment + dataset
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
```
## 💾 Datasets Bag Files
Google drive link: https://drive.google.com/drive/folders/1LHEkOGGjcsvM41ybrqrOnuIwgvTVOZE_?usp=sharing

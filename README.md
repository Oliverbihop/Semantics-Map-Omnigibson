<p align="center">
  <!-- Replace with your own banner image if available -->
  <img src="[https://via.placeholder.com](https://github.com/Oliverbihop/Semantics-Map-Omnigibson/blob/main/assets/map_capture.png)/800x200?text=Semantics-Map-Omnigibson" alt="Project Banner" />
</p>

<h1 align="center">Semantics-Map-Omnigibson</h1>

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

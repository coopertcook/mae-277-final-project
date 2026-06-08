# MAE 277 Final Project

## Cloning with Submodules

This repository uses Git submodules. Follow the steps below to set up correctly.

### Option 1: Fresh Clone (recommended)

```bash
git clone --recurse-submodules git@github.com:UC-BIRD-Lab/<repo-name>.git
```

This clones the repo and initializes all submodules in one step.

### Option 2: Already Cloned Without Submodules

If you already cloned the repo and the submodule folders are empty, run:

```bash
git submodule update --init --recursive
```

## SSH Setup (Required)

The submodules use SSH URLs, so you need an SSH key configured for GitHub.

### 1. Generate an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### 2. Add your public key to GitHub

- Go to **GitHub → Settings → SSH and GPG keys → New SSH key**
- Paste the contents of `~/.ssh/id_ed25519.pub`

### 3. Add your key to the SSH agent

```bash
# Windows (PowerShell)
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519

# macOS / Linux
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 4. Test the connection

```bash
ssh -T git@github.com
# Expected: Hi <username>! You've successfully authenticated...
```

### Troubleshooting: Git uses wrong SSH on Windows

If `ssh -T git@github.com` works but `git submodule update` still fails, git may be using its own bundled SSH. Fix it by pointing git to the system SSH:

```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"
```

Then retry `git submodule update --init --recursive`.

## Environment Setup

### Linux / macOS

A conda environment file is provided:

```bash
conda env create -f environment.yaml
conda activate mae-277
```

### Windows

The `environment.yaml` was built on Linux and cannot be used directly on Windows. Use the provided `requirements.txt` instead:

```powershell
conda create -n mae-277 python=3.13
conda activate mae-277
pip install -r requirements.txt
```

> **Note:** `triton`, `pexpect`, and `ptyprocess` are Linux-only and have been excluded from `requirements.txt`. If you encounter missing package errors, install them manually with `pip install <package>`.

> **GPU (CUDA) packages** such as `torch` with CUDA support may require a separate install step depending on your NVIDIA driver version. See [PyTorch installation guide](https://pytorch.org/get-started/locally/) if needed.

## Installing the GIFT Submodule as a Package

After cloning and activating your environment, install the `generalized-interpolative-flight-tool` submodule as an editable package so Python can find it:

```bash
pip install -e generalized-interpolative-flight-tool
```

This only needs to be done once. The `-e` flag means it points directly to the submodule folder, so any changes there are reflected immediately without reinstalling.

## Running the Simulation

```bash
python run_simulation.py
```

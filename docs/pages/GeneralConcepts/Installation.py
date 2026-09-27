# # Installing MTEX for Python
#
# MATLAB comes as one program: the language, the editor, the command window and the figure
# windows are installed together. In Python these are separate pieces, and this page
# installs them one after the other:
#
# 1. **Python**, the language that runs your scripts,
# 2. **VS Code**, the editor in which you write and run them, playing the part of the MATLAB
#    desktop,
# 3. **MTEX**, installed into a folder of its own for your project.
#
# It takes about fifteen minutes and needs an internet connection. No previous knowledge of
# Python is assumed; if you already work with Python, the short version is
# `pip install "mtex[all]"` into an environment with Python 3.12 or later.
#
# ## Step 1: Install Python
#
# Install **Python 3.14**. Any version from 3.12 on works, but take 3.14 even when a newer
# one is offered: a Python version released only a few months ago often lacks the compiled
# packages MTEX builds on.
#
# **Windows.** Open [python.org/downloads](https://www.python.org/downloads/) and click
# *get the standalone installer for Python 3.14*. Run the downloaded file, and on its first
# screen tick **Add python.exe to PATH** before clicking *Install Now*. Without that tick,
# the other programs will not find Python.
#
# **macOS.** Open [python.org/downloads](https://www.python.org/downloads/), download the
# macOS installer of Python 3.14 and run it. When it has finished, a Finder window with the
# new Python folder opens; double-click **Install Certificates.command** in it. Without this
# step, Python cannot download the MTEX sample data.
#
# **Linux.** Most distributions bring Python. Open a terminal and type `python3 --version`.
# A version of 3.12 or later is fine; on Debian and Ubuntu also install the package that
# creates environments, `sudo apt install python3-venv`. With an older Python, install a
# current one with [uv](https://docs.astral.sh/uv/), see *Other ways* below.
#
# To check the installation, open a terminal (on Windows: press the Windows key, type
# `cmd` and press Enter; on macOS: the program *Terminal*) and type
#
# ```
# python --version
# ```
#
# on Windows, or `python3 --version` on macOS and Linux. It should answer `Python 3.14.x`.
#
# ## Step 2: Install VS Code
#
# Download Visual Studio Code from [code.visualstudio.com](https://code.visualstudio.com/)
# and install it with its default settings. VS Code is a general editor; Python support
# comes as extensions:
#
# 1. Start VS Code and click the *Extensions* icon in the bar on the left (four small
#    squares), or press Ctrl+Shift+X (Cmd+Shift+X on macOS).
# 2. Search for **Python** and install the extension published by Microsoft.
# 3. Search for **Jupyter** and install the extension published by Microsoft.
#
# The first lets VS Code run Python, the second gives the interactive window in which a
# script runs piece by piece and shows its figures, as MATLAB's editor does with its
# sections.
#
# ## Step 3: A project folder with MTEX
#
# Python keeps the packages a project uses in an **environment**: a folder, called `.venv`,
# inside the project folder, holding its own copy of MTEX and everything MTEX needs. It is
# the counterpart of adding MTEX to the MATLAB path, but per project, so a project keeps
# working when another one installs a different version. You create the environment once
# per project folder.
#
# 1. Create a folder for your analysis, for instance `Documents/mtex-first-steps`.
# 2. In VS Code choose *File > Open Folder...* and open it. If VS Code asks whether you trust
#    the authors of the folder, answer yes.
# 3. Press Ctrl+Shift+P (Cmd+Shift+P on macOS), type **Python: Create Environment** and
#    choose it. Choose **Venv**, then the Python 3.14 you installed in step 1. A folder
#    `.venv` appears in the file list on the left.
# 4. Open a terminal inside VS Code with *Terminal > New Terminal*. Its prompt starts with
#    `(.venv)`, which says the environment is active. Type
#
# ```
# pip install "mtex[all]" ipykernel
# ```
#
# and wait until pip reports *Successfully installed*. This downloads MTEX and about 300 MB
# of packages it builds on. The quotes matter: without them, some terminals misread the
# square brackets.
#
# `mtex[all]` installs MTEX with every optional part: the compiled kernels (numba), the
# plotting (matplotlib), the HDF5 file formats (h5py), the 3D scenes (PyVista) and the
# import wizard (PySide6). A bare `pip install mtex` installs the computations only.
# `ipykernel` is what the interactive window runs your code with.
#
# ## Step 4: Your first script
#
# Choose *File > New File...*, then *Python File*, and save it as `first.py` in the project
# folder. Type
#
# ```python
# # %%
# from mtex import *
#
# # %%
# ebsd = mtexdata('forsterite')
# ebsd
#
# # %%
# plot(ebsd)
# ```
#
# A line `# %%` starts a **cell**, as `%%` starts a section in the MATLAB editor. Click into
# the first cell and press **Shift+Enter**: the cell runs in the *Interactive* window that
# opens on the right, and the cursor moves to the next cell. The first time, VS Code asks
# for a kernel; choose the one named `.venv`. Press Shift+Enter twice more. The second cell
# downloads a sample EBSD map and shows its summary, the third draws the phase map.
#
# The first run of many MTEX commands takes a few seconds longer than later ones: MTEX
# compiles its fast routines on first use and keeps them on disk for the next time.
#
# A few habits help:
#
# * **Variables.** The *Variables* button in the toolbar of the interactive window lists
#   the variables and their sizes, like MATLAB's workspace.
# * **Run everything.** *Run All* in the toolbar above the script runs every cell. The
#   triangle at the top right, *Run Python File*, runs the script in the terminal instead;
#   there it shows no figures, so use it only for scripts that save their results, for
#   instance with `saveFigure('map.png')`.
# * **Figures in windows.** The interactive window shows figures as pictures. To get
#   separate figure windows that zoom and turn, as in MATLAB, run `%matplotlib qt` once in
#   the interactive window.
# * **Start afresh.** *Restart* in the toolbar of the interactive window clears all
#   variables, like MATLAB's `clear all`.
#
# ## The Python prompt: MATLAB's command window
#
# In MATLAB you type a command into the command window, press Enter and see the answer. In
# VS Code the same place is the input box at the bottom of the interactive window, marked
# *Type 'python' code here and press Shift+Enter to run*.
#
# 1. Click into the input box.
# 2. Type a command, for instance `ebsd['Forsterite']`.
# 3. Press **Shift+Enter**. A plain Enter starts a second line instead, so that a command
#    can span several lines.
#
# The command runs with the variables of the cells you ran before, so after the first
# script `ebsd` is known here, and a variable you define here is known in the next cell of
# the script. The answer, or the figure, appears above the input box. In the box, the up
# arrow brings back the commands you typed earlier, and typing `ebsd.` followed by Ctrl+Space
# lists what `ebsd` offers.
#
# To get a prompt without opening a script first, press Ctrl+Shift+P (Cmd+Shift+P on macOS)
# and choose **Jupyter: Create Interactive Window**. A new interactive window opens with its
# input box; choose the kernel named `.venv` if VS Code asks, and start with
#
# ```python
# from mtex import *
# ```
#
# The terminal at the bottom of VS Code is a second kind of prompt: it runs system commands
# such as `pip install`, not Python. Typing `python` there starts a plain Python prompt,
# marked `>>>`, which runs Python but shows no figures; leave it with `exit()`. Use the
# interactive window for MTEX and the terminal for installing.
#
# ## Working through the documentation
#
# Every page of this documentation is a script: the text you read, and the code between it,
# which produced every number and figure on the page. The best way to learn MTEX is to run a
# page yourself, section by section, and change it. MATLAB MTEX users know this from opening
# a documentation script in the MATLAB editor.
#
# **Get the page as a script.** In the interactive window type
#
# ```python
# openDoc('EBSDTutorial')
# ```
#
# This writes the page *EBSD Tutorial* as the file `EBSDTutorial.py` into your project
# folder and opens it in the editor. The name of a page is the last part of its web address
# without `_py.html`: the page `.../EBSDTutorial_py.html` is `EBSDTutorial`. `openDoc()`
# without a name lists every page by chapter, and `openDoc('Tutorials')` the pages of one
# chapter. Instead of `openDoc` you can also click *download as script* at the top of a page
# on the website and save the file into your project folder.
#
# If the file does not open by itself (on macOS VS Code needs *Shell Command: Install 'code'
# command in PATH* from Ctrl+Shift+P for that), click it in the file list on the left.
#
# **Read the script.** The page is cut into cells, each starting with a line `# %%`:
#
# * a cell marked `# %% [markdown]` is text of the page, its lines starting with `#`,
# * a cell marked `# %%` alone is code,
# * above each cell VS Code shows the small links *Run Cell*, *Run Above* and *Debug Cell*.
#
# **Run it cell by cell.** Click into the first cell and press **Shift+Enter** again and
# again. Every press runs one cell in the interactive window and moves on to the next: a text
# cell appears there as formatted text, a code cell prints its results and draws its figures.
# So the interactive window rebuilds the page as you go, and you can compare it with the page
# on the website. The first cell that needs sample data downloads it, which takes a moment
# once.
#
# **Keep the order.** A cell uses the variables of the cells above it. When you jump ahead
# and a cell fails with `NameError: name 'ebsd' is not defined`, a cell above it has not
# run: click *Run Above* on the cell, then run it again.
#
# **Change things.** The script is your copy; edit a number, an option or a colour and press
# **Ctrl+Enter**, which runs the cell again without moving on. To see the effect of a change
# on everything below, continue with Shift+Enter from there. `openDoc` never overwrites a
# script you have already, so your changes are safe; `openDoc('EBSDTutorial',
# overwrite=True)` brings back the original.
#
# **Start over.** *Restart* in the toolbar of the interactive window forgets all variables;
# run the cells from the top again.
#
# **Where to start.** The chapter *Tutorials* goes through a whole analysis:
# [EBSD Tutorial](EBSDTutorial.html), [Grain Tutorial](GrainTutorial.html),
# [Grain Boundary Tutorial](BoundaryTutorial.html), then
# [Pole Figure Tutorial](PoleFigureTutorial.html) and [ODF Tutorial](ODFTutorial.html). For
# the ideas behind the commands continue with [MTEX Scripts](MTEXScripts.html). If you know
# MTEX from MATLAB, read [MTEX in MATLAB and in Python](MATLABtoPython.html) first.
#
# ## The import wizard
#
# The import wizard opens EBSD files in a window and writes the Python script that imports
# them; see [EBSD Import](EBSDImport.html). Start it in the terminal of VS Code, with the
# environment active, by
#
# ```
# mtex-wizard
# ```
#
# or from a script with `importWizard()`.
#
# ## Updating MTEX
#
# In the terminal of VS Code, with `(.venv)` in the prompt, type
#
# ```
# pip install --upgrade "mtex[all]"
# ```
#
# and restart the interactive window. `pip show mtex` tells the version you have.
#
# ## A second project
#
# A new project folder needs its own environment: repeat step 3 there. To use one
# environment for several folders instead, choose *Python: Select Interpreter* from
# Ctrl+Shift+P in the new folder and pick *Enter interpreter path...*, then the file
# `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS, Linux) of the first
# project.
#
# ## When something goes wrong
#
# **`python` is not recognized, or opens the Microsoft Store (Windows).** The tick *Add
# python.exe to PATH* was missed. Run the installer again, choose *Modify*, then *Next*, and
# tick *Add Python to environment variables*. Close and reopen every terminal and VS Code
# afterwards.
#
# **`ModuleNotFoundError: No module named 'mtex'`.** The code runs in another Python than
# the one MTEX was installed into. For the interactive window, click the kernel name at its
# top right and choose `.venv`. For the terminal, look at the bottom right of the VS Code
# window: it names the Python in use; click it and choose the one in `.venv`.
#
# **The terminal refuses to activate the environment: *running scripts is disabled on this
# system* (Windows).** Windows forbids the activation script by default. Type once
#
# ```
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ```
#
# in the terminal of VS Code, confirm, and open a new terminal.
#
# **`CERTIFICATE_VERIFY_FAILED` when `mtexdata` downloads (macOS).** The step *Install
# Certificates.command* of step 1 was skipped. Open the folder *Applications > Python 3.14*
# and double-click it.
#
# **pip fails while building `finufft`.** finufft, which MTEX uses for its fast Fourier
# transforms, ships ready compiled for Windows on Intel and AMD processors, macOS 14 or later
# on Apple silicon and Linux on Intel and AMD processors. On other machines (an Intel Mac,
# Windows on ARM) pip tries to compile it, which needs a C++ compiler and FFTW; the
# [finufft installation notes](https://finufft.readthedocs.io/en/latest/install.html) explain
# how.
#
# **The interactive window asks to install `ipykernel`.** It was left out of step 3. Click
# *Install*, or type `pip install ipykernel` in the terminal.
#
# ## Other ways
#
# Nothing on this page is tied to VS Code: MTEX runs in any Python environment, and the
# same `pip install "mtex[all]"` works in Spyder, PyCharm, JupyterLab or a plain terminal.
#
# **Anaconda or Miniforge.** Create an environment and install MTEX into it with pip:
#
# ```
# conda create -n mtex python=3.14
# conda activate mtex
# pip install "mtex[all]" ipykernel
# ```
#
# In VS Code choose the interpreter or kernel named `mtex`.
#
# **uv.** [uv](https://docs.astral.sh/uv/) installs Python and the packages in one tool. In
# the project folder:
#
# ```
# uv init --python 3.14
# uv add "mtex[all]" ipykernel
# ```
#
# **The newest MTEX from GitHub.** Install from the repository instead of the release:
#
# ```
# pip install "mtex[all] @ git+https://github.com/mtex-toolbox/pymtex"
# ```

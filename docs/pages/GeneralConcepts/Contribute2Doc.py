# # Contributing to the MTEX documentation
#
# Useful contributions to this documentation include spelling corrections, worked examples,
# theoretical explanations and special use cases.
#
# Each help page is an executable Python script. A good change must therefore improve the
# explanation and leave the example runnable. The MATLAB documentation of MTEX is the
# reference of every page: its text, the order of its examples, the numbers it quotes and the
# figures it shows. A page of this port is that page with the Python code under it.
#
# ## Where a page lives
#
# A page is one script `docs/pages/<Chapter>/<Page>.py`, named as the MATLAB script
# `doc/<Chapter>/<Page>.m` is. It is written in the
# [jupytext light format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format):
#
# * a comment block preceded by a blank line is prose, Markdown with `$...$` maths;
# * the code between two prose blocks is one cell, and its output is shown under it;
# * `# +` and `# -` enclose a cell that contains blank lines;
# * `# ---` ends a cell without prose.
#
# A page begins with its title as a `# # Title` heading and with the imports it uses,
# `from mtex import ...` and `import numpy as np`, and sets
# `plottingConvention.default('y↑→x')` where the MATLAB page sets it. Use backticks for
# inline code, as in `calcGrains`, and include the `.html` suffix in internal links, as in
# [Clustering](ClusterDemo.html).
#
# ## Write the page from the MATLAB page
#
# Keep the MATLAB text paragraph for paragraph and the examples in its order, and write the
# Python code as short as the MATLAB code. When the Python code of an example comes out
# longer than the MATLAB code, the port lacks a verb, a property or an option: add it to the
# port with a test instead of working around it on the page. Numbers the MATLAB page quotes
# stay in the text, so that a differing Python result shows when the page is read.
#
# A page ends with a **Technical Details** section only when there is something to say: what
# differs from MATLAB and why, what the port derives rather than stores, what is still owed.
#
# ## Make figures earn their place
#
# A generated figure belongs immediately after the code that creates it. Add a sentence that
# tells the reader what feature or comparison to notice.
#
# Use a static PNG or SVG only when the page cannot draw the concept clearly. Keep the file
# beside the page in `docs/pages/<Chapter>/`, from where the build copies it.
#
# ## Build and check the page
#
# The build executes a page cell by cell in one namespace, as MATLAB's `publish` does:
#
#     .venv/bin/python docs/make.py <Page>
#
# It writes `docs/build/<Page>.md` with the code of every cell, its output and its figures,
# `<Page>-<cell>.png`. A failing cell stops the page and the script prints the traceback.
# `docs/notebook.py <Page>` opens the page as a notebook to run it section by section.
#
# A page is checked against the MATLAB run of the same page: `docs/matlabrun.py
# <Chapter>/<Page>` runs the MATLAB script into `docs/matlab/<Page>/`, the diary with a
# marker per cell and one PNG per figure. Read the diary against the output blocks number
# for number, and the figures by eye; `docs/montage.py <Page>` puts the MATLAB figures beside
# the built ones in `docs/build/_montage_<Page>.png`. Where the numbers cannot agree, random
# draws or a feature only MATLAB has, the page says so.
#
# ## Submit a reviewable change
#
# A documentation change records its page's status in `docs/plans/remaining-pages.md`
# (Checked, Adapted, Partial, Written) and what the page found, a bug or a difference from
# MATLAB, in `docs/findings.md`. It should say what confused the reader, summarize the
# correction and list the build or the comparison used as evidence. Include a
# before-and-after image when the rendered figure changes.
#
# If the documentation accompanies a change of the port, the change carries its tests in
# `tests/test_<file>.py`. The plan of the documentation, how a page is shaped, built and
# checked, is `docs/plans/docs.md`.
#
# ## Technical Details
#
# The MATLAB page describes MTEX's own route: editing the `.m` source on GitHub, previewing
# it with MATLAB's `publish` and `makeDoc`, the structural checker
# `doc/tools/check_doc_structure.py`, and a pull request against the `develop` branch of
# `mtex-toolbox/mtex`. A correction to the text both ports share belongs there first.

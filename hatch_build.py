# the documentation pages go into the wheel as mtex/docpages, their scripts only, for openDoc
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
  def initialize(self, version, build_data):
    for f in sorted((Path(self.root) / "docs" / "pages").glob("*/*.py")):
      build_data["force_include"][str(f)] = f"mtex/docpages/{f.parent.name}/{f.name}"

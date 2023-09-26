from chalice import Chalice
import importlib
from pathlib import Path
import sys
from dataclasses import dataclass, KW_ONLY
from typing import Container, Optional, Union, Literal
import os

PathType = Union[str, os.PathLike]
AutoInitOptions = Literal[False, 0, True, 1, 2]

app = Chalice(app_name="pfun-cma-model-deps")

sitepkgs_path = Path("/opt/python/lib/python%s.%s/site-packages" %
                     sys.version_info[:2])
depsfile_path = (sitepkgs_path.joinpath("chalicelib",
                                        "dependencies_latest.txt")
                 if sitepkgs_path.joinpath("chalicelib").exists() else
                 Path(".").joinpath("dependencies_latest.txt").resolve())


@dataclass
class DependenciesInfo:
    sitepkgs_path: PathType = sitepkgs_path
    depsfile_path: PathType = depsfile_path
    depslist: Optional[Container[str]] = None
    # trunk-ignore(ruff/B018)
    KW_ONLY
    deps_str: Optional[str] = None
    autoinit_depslist: Optional[AutoInitOptions] = 2

    def __post_init__(self,
                      *args,
                      depslist=None,
                      deps_str=None,
                      autoinit_depslist: AutoInitOptions = 2):
        self.depslist = depslist
        self.deps_str = deps_str
        #: ! only init depslist if the path doesn't resolve (when autoinit < 2)
        if int(autoinit_depslist) < 2:
            autoinit_depslist = all([
                int(autoinit_depslist) > 0,
            ])
        else:
            autoinit_depslist = int(autoinit_depslist) > 0
        if self.depslist is None and autoinit_depslist is True:
            self.depslist = self.make_depslist(self.depsfile_path)
        if self.deps_str is None:
            #: ! should fail if self.depslist is still None
            self.deps_str = "\n".join(self.depslist)

    def make_depslist(self, cache: bool = True):
        depslist = []

        def check_dep(dep: str):
            if len(dep) == 0:
                return False
            return all(["_" != dep[0], "__" not in dep])

        for p in self.sitepkgs_path.iterdir():
            if any([
                    not p.is_dir(), (".dist-info" not in p.suffix), "chalice"
                    in p.name
            ]):
                continue
            try:
                dlist = (p.joinpath("top_level.txt").resolve(
                    strict=True).read_text().split("\n"))
            except FileNotFoundError:
                continue
            depslist.extend([dep for dep in dlist if check_dep(dep)])
        if cache is True:
            self.depslist = depslist
        return depslist


def init_depfile():
    depsobj = DependenciesInfo()
    return depsobj


@app.lambda_function()
def import_dependencies(event, context):
    depsobj = init_depfile()
    modules = []
    for dependency in depsobj.depslist:
        module = importlib.import_module(dependency)
        modules.append({"name": dependency, "__file__": module.__file__})
    return modules

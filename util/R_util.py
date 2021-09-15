import sys

from rpy2.robjects.packages import importr, isinstalled
import rpy2.robjects as ro
from rpy2.robjects import FloatVector, IntVector
import rpy2.robjects.numpy2ri


def activate_robjects():
    rpy2.robjects.numpy2ri.activate()


def convert_object_to_matrix(m):
    nrow, ncol = m.shape
    return ro.r.matrix(m, nrow=nrow, ncol=ncol)


def convert_list_to_intVector(l):
    return IntVector(l)


def convert_list_to_floatVector(l):
    return FloatVector(l)


def import_packages(pkgs):
    for pkg in pkgs:
        import_package(pkg=pkg)


def import_package(pkg):
    importr(pkg)


def install_packages(pkgs, mirror=0):
    utils = importr('utils')
    utils.chooseCRANmirror(ind=mirror)
    for pkg in pkgs:
        # if True:
        utils.install_packages(pkg)


def install_package(pkg, mirror=0, install_type=sys.platform):
    utils = importr('utils')
    utils.chooseCRANmirror(ind=mirror)
    if not isinstalled(pkg):
        if install_type == "source":
            utils.install_packages(pkg, repos=rpy2.robjects.NULL, type=install_type, dep=True, dependencies=True)
        else:
            utils.install_packages(pkg)


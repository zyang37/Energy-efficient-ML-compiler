# did not work:

# from pyJoules.energy_meter import measure_energy
# from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain
#
# @measure_energy(domains=[RaplPackageDomain(1)])
# def foo():
#   for i in range(100 * 1000):
#     j = 69.234 * 420.376259 * i
#
#
# foo()

# pyRAPL does not work either
# import pyRAPL
#
# pyRAPL.setup()
#
# @pyRAPL.measure
# def foo():
#   for i in range(100 * 1000):
#     j = 69.234 * 420.376259 * i
#
#
# foo()
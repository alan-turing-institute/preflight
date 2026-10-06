# HOSTGUARD-SHIM: this directory on PYTHONPATH loads the guard (hostguard.py, same dir).
import hostguard

hostguard.chain_shadowed_sitecustomize()

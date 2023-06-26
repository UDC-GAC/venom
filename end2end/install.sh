pip uninstall -y spatha
rm -rf build
rm -rf dist
rm -rf spatha.egg-info
python3 -W ignore setup.py build
python3 -W ignore setup.py install
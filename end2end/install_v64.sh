pip uninstall -y spatha
rm -rf build
rm -rf dist
rm -rf spatha.egg-info
python3 -W ignore setup_v64.py build
python3 -W ignore setup_v64.py install
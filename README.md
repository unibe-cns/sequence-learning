# sequence-learning

## instructions to contribute

1. fork this repo
2. clone the fork: `git clone git@github.com:USERNAME/sequence-learning.git`
3. add the upstream to local clone: `git remote add upstream git@github.com:unibe-cns/sequence-learning.git`
4. create a python environment, 
  - either with venv: `python -m venv --system-site-packages <name_of_env>` and activate it: `source ./<name_of_env>/bin/activate`
  - or with conda: `conda create -n <name_of_env>` (maybe add the python version: `python=3.9`) and activate it: `conda activate <name_of_env>`
5. install the dependies: `python -m pip install -r requirements.txt`
6. install this very pip-package: `python -m pip install -e .`
7. install the pre-commit hooks: `pre-commit install`
8. happy coding! 

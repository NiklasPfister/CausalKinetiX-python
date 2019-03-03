import setuptools

setup(
    name='CausalKinetiX',
    version='0.0.1',
    description='Python implementation of CausalKinetiX (originally developed in R).',
    long_description=readme,
    author='Niklas Pfister, Stefan Bauer, Jonas Peters, Kei Ishikawa',
    author_email='niklas.pfister@stat.math.ethz.ch, stefan.bauer@inf.ethz.ch, jonas.peters@math.ku.dk, ishikawa-kei521@g.ecc.u-tokyo.ac.jp',
    url='https://github.com/NiklasPfister/CausalKinetiX-python',
    install_requires=['numpy','scipy','quadprog','matplotlib', 'copy','glmnet','itertools']
    #license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
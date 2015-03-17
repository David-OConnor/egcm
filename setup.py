from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name = "egcm",
    version = "0.1",
    packages = find_packages(),
    # packages = ['egcm',
    #             ],

    install_requires = ['arch>=3.0',
                        'arrow>=0.5.4',
                        'numpy>=1.9.1',
                        'pandas>=0.15.2',
                        'statsmodels>=0.6.1'],

    package_dir = {'egcm': 'egcm'},

    author = "Matthew Clegg, David O'Connor",
    author_email = "david.alan.oconnor@gmail.com",
    description = "Engle-Granger 2-step Cointegration for Python",
    long_description = readme,
    license = "LGPL",
    keywords = "cointegration, egcm, engle-granger",

)
